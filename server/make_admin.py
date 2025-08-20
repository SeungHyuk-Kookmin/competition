import os, sys
from sqlmodel import Session, select, create_engine

try:
    # 패키지 실행 시
    from server.app import User, DB_URL, engine
except Exception:
    # CWD가 server/일 때
    sys.path.append(os.path.dirname(__file__))                 # /server
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))# / (project root)
    from app import User
    DB_URL = os.getenv("DATABASE_URL", "sqlite:///./leaderboard.db")
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})

def set_admin(email: str, on: bool=True):
    with Session(engine) as s:
        u = s.exec(select(User).where(User.email == email.strip().lower())).first()
        if not u:
            print(f"❌ user not found: {email}"); sys.exit(1)
        u.is_admin = bool(on)
        s.add(u); s.commit()
        print(f"✅ set is_admin={u.is_admin} for {u.email}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--email", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--on", action="store_true")
    g.add_argument("--off", action="store_true")
    args = p.parse_args()
    set_admin(args.email, on=(not args.off))
