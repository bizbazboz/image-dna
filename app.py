from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import imagehash
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_PATH = Path(os.getenv("IMAGE_HASH_DATABASE", "image_hashes.db"))

# pHash with hash_size=8 produces an 8 × 8 = 64-bit hash.
HASH_SIZE = 8
HASH_BITS = HASH_SIZE * HASH_SIZE

MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_IMAGE_PIXELS = 40_000_000

DEFAULT_MATCH_THRESHOLD = 8

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    id: int
    filename: str
    content_type: str | None
    perceptual_hash: str
    width: int
    height: int
    hash_bits: int
    exact_hashes_already_stored: int


class MatchResult(BaseModel):
    id: int
    filename: str
    content_type: str | None
    perceptual_hash: str
    distance: int
    confidence: float
    similarity_percent: float
    width: int
    height: int
    created_at: str


class CompareResponse(BaseModel):
    query_hash: str
    total_hashes: int
    matches: list[MatchResult]


class CheckResponse(BaseModel):
    appears_in_database: bool
    query_hash: str
    threshold: int
    confidence: float
    best_match: MatchResult | None


class HealthResponse(BaseModel):
    status: str
    stored_hashes: int


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def hamming_distance(hash_a: str, hash_b: str) -> int:
    """
    Calculate the number of differing bits between two hexadecimal hashes.
    """
    try:
        return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()
    except (TypeError, ValueError):
        return HASH_BITS


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")

    connection.create_function(
        "hamming_distance",
        2,
        hamming_distance,
        deterministic=True,
    )

    return connection


def initialise_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        # WAL improves read/write behaviour when multiple requests arrive.
        connection.execute("PRAGMA journal_mode = WAL")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content_type TEXT,
                perceptual_hash TEXT NOT NULL,
                hash_bits INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Useful for exact-hash lookups.
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_image_hashes_perceptual_hash
            ON image_hashes(perceptual_hash)
            """
        )


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

async def read_uploaded_image(upload: UploadFile) -> bytes:
    """
    Read an uploaded image while enforcing a maximum size.
    """
    data = await upload.read(MAX_UPLOAD_SIZE + 1)

    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"The image must not exceed {MAX_UPLOAD_SIZE // 1024 // 1024} MB.",
        )

    return data


def calculate_image_hash(data: bytes) -> tuple[str, int, int]:
    """
    Decode an image and calculate its 64-bit perceptual hash.
    """
    try:
        with Image.open(BytesIO(data)) as source_image:
            # Use the first frame for animated images.
            source_image.seek(0)

            # Normalise phone-camera EXIF rotation before hashing.
            corrected_image = ImageOps.exif_transpose(source_image)
            corrected_image.load()

            width, height = corrected_image.size

            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions.")

            rgb_image = corrected_image.convert("RGB")

            perceptual_hash = imagehash.phash(
                rgb_image,
                hash_size=HASH_SIZE,
            )

            return str(perceptual_hash), width, height

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid or supported image.",
        ) from error


def safe_filename(filename: str | None) -> str:
    """
    Remove any client-provided directory components.
    """
    if not filename:
        return "unnamed-image"

    return Path(filename.replace("\\", "/")).name or "unnamed-image"


def distance_to_confidence(distance: int, hash_bits: int = HASH_BITS) -> float:
    """
    Convert Hamming distance into a normalised 0–1 similarity value.

    This is a similarity score, not a statistical probability.
    """
    if hash_bits <= 0:
        return 0.0

    score = 1.0 - (distance / hash_bits)
    return round(max(0.0, min(1.0, score)), 4)


def row_to_match(row: sqlite3.Row) -> MatchResult:
    distance = int(row["distance"])
    hash_bits = int(row["hash_bits"])
    confidence = distance_to_confidence(distance, hash_bits)

    return MatchResult(
        id=row["id"],
        filename=row["filename"],
        content_type=row["content_type"],
        perceptual_hash=row["perceptual_hash"],
        distance=distance,
        confidence=confidence,
        similarity_percent=round(confidence * 100, 2),
        width=row["width"],
        height=row["height"],
        created_at=row["created_at"],
    )


def find_similar_hashes(
    query_hash: str,
    limit: int,
) -> tuple[int, list[MatchResult]]:
    """
    Compare a hash against every stored hash and return the closest results.

    SQLite invokes the registered Python hamming_distance() function for each
    database row, then orders the results by distance.
    """
    with get_connection() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM image_hashes"
        ).fetchone()[0]

        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                content_type,
                perceptual_hash,
                hash_bits,
                width,
                height,
                created_at,
                hamming_distance(perceptual_hash, ?) AS distance
            FROM image_hashes
            ORDER BY distance ASC, id ASC
            LIMIT ?
            """,
            (query_hash, limit),
        ).fetchall()

    return total, [row_to_match(row) for row in rows]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    initialise_database()
    yield


app = FastAPI(
    title="Perceptual Image Hash API",
    description=(
        "Store perceptual image hashes and compare uploaded images "
        "against hashes stored in SQLite."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "message": "Perceptual Image Hash API",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health() -> HealthResponse:
    with get_connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM image_hashes"
        ).fetchone()[0]

    return HealthResponse(
        status="healthy",
        stored_hashes=count,
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Images"],
)
async def upload_image(
    image: Annotated[
        UploadFile,
        File(description="Image whose perceptual hash should be stored"),
    ],
) -> UploadResponse:
    image_data = await read_uploaded_image(image)
    perceptual_hash, width, height = calculate_image_hash(image_data)

    filename = safe_filename(image.filename)

    with get_connection() as connection:
        existing_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM image_hashes
            WHERE perceptual_hash = ?
            """,
            (perceptual_hash,),
        ).fetchone()[0]

        cursor = connection.execute(
            """
            INSERT INTO image_hashes (
                filename,
                content_type,
                perceptual_hash,
                hash_bits,
                width,
                height
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                image.content_type,
                perceptual_hash,
                HASH_BITS,
                width,
                height,
            ),
        )

        record_id = cursor.lastrowid

    if record_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The image hash could not be stored.",
        )

    return UploadResponse(
        id=record_id,
        filename=filename,
        content_type=image.content_type,
        perceptual_hash=perceptual_hash,
        width=width,
        height=height,
        hash_bits=HASH_BITS,
        exact_hashes_already_stored=existing_count,
    )


@app.post(
    "/compare",
    response_model=CompareResponse,
    tags=["Images"],
)
async def compare_image(
    image: Annotated[
        UploadFile,
        File(description="Image to compare with stored hashes"),
    ],
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description="Maximum number of similar results to return",
        ),
    ] = 5,
) -> CompareResponse:
    image_data = await read_uploaded_image(image)
    query_hash, _, _ = calculate_image_hash(image_data)

    total, matches = find_similar_hashes(
        query_hash=query_hash,
        limit=limit,
    )

    return CompareResponse(
        query_hash=query_hash,
        total_hashes=total,
        matches=matches,
    )


@app.post(
    "/check",
    response_model=CheckResponse,
    tags=["Images"],
)
async def check_image(
    image: Annotated[
        UploadFile,
        File(description="Image to check against the database"),
    ],
    threshold: Annotated[
        int,
        Query(
            ge=0,
            le=HASH_BITS,
            description=(
                "Maximum Hamming distance considered a match. "
                "Lower values are stricter."
            ),
        ),
    ] = DEFAULT_MATCH_THRESHOLD,
) -> CheckResponse:
    image_data = await read_uploaded_image(image)
    query_hash, _, _ = calculate_image_hash(image_data)

    _, matches = find_similar_hashes(
        query_hash=query_hash,
        limit=1,
    )

    if not matches:
        return CheckResponse(
            appears_in_database=False,
            query_hash=query_hash,
            threshold=threshold,
            confidence=0.0,
            best_match=None,
        )

    best_match = matches[0]
    appears = best_match.distance <= threshold

    return CheckResponse(
        appears_in_database=appears,
        query_hash=query_hash,
        threshold=threshold,
        confidence=best_match.confidence,
        best_match=best_match,
    )