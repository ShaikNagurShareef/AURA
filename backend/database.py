"""
AURA – Database Setup (SQLAlchemy + SQLite)
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config import settings

# connect_args only needed for SQLite
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Called at startup to create all tables."""
    # Import models so SQLAlchemy registers them before create_all
    import backend.models.user  # noqa: F401
    import backend.models.session  # noqa: F401
    import backend.models.message  # noqa: F401

    Base.metadata.create_all(bind=engine)
