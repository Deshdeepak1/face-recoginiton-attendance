import databases
import ormar
import sqlalchemy
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

database = databases.Database(settings.db_url)
metadata = sqlalchemy.MetaData()


class BaseMeta(ormar.ModelMeta):
    metadata = metadata
    database = database


class User(ormar.Model):
    class Meta(BaseMeta):
        tablename = "users"

    id: int = ormar.Integer(primary_key=True)
    name: str = ormar.String(max_length=128, nullable=False)
    email: str = ormar.String(max_length=128, unique=True, nullable=False)
    filename: str = ormar.String(max_length=128, unique=True, nullable=False)


engine = sqlalchemy.create_engine(settings.db_url)
try:
    metadata.create_all(engine)
except Exception:
    pass

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
