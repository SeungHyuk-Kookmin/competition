import os, sys
from sqlmodel import SQLModel, create_engine, Session, select

# --- flexible import ---
try:
    from server.app import User, hash_password
except ModuleNotFoundError:
    try:
        from app import User, hash_password
    except ModuleNotFoundError:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from server.app import User, hash_password

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./leaderboard.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SQLModel.metadata.create_all(engine)

EMAIL = os.environ.get("ADMIN_EMAIL", "chungii2287@gmail.com")
TEAM  = os.environ.get("ADMIN_TEAM", "Admins")
PWD   = os.environ.get("ADMIN_PASSWORD", "Tkdhak6708!")

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