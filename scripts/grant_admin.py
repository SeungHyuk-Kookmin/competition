import os
from typing import Optional
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy import UniqueConstraint
from passlib.context import CryptContext

# 동일 스키마의 User 모델 (server.app 임포트 없이)
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str
    team: str
    password_hash: str
    is_admin: bool = False
    __table_args__ = (
        UniqueConstraint("email", name="uix_user_email"),
        UniqueConstraint("team", name="uix_user_team"),
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)

# DB 연결
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./leaderboard.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SQLModel.metadata.create_all(engine)  # 기존 테이블 있으면 그대로 둠

EMAIL = os.environ.get("ADMIN_EMAIL", "admin@example.com").strip().lower()
TEAM  = os.environ.get("ADMIN_TEAM", "Admin").strip()
PWD   = os.environ.get("ADMIN_PASSWORD", "admin123")

with Session(engine) as s:
    u = s.exec(select(User).where(User.email == EMAIL)).first()
    if u:
        u.is_admin = True
        s.add(u); s.commit()
        print(f"Promoted to admin: {EMAIL} (team={u.team})")
    else:
        u = User(email=EMAIL, team=TEAM, password_hash=hash_password(PWD), is_admin=True)
        s.add(u); s.commit()
        print(f"Created admin: {EMAIL} / {PWD} (team={TEAM})")