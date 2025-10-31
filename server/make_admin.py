# make_admin.py
import os
import sys

# ---- 경로 보정 ----
# 현재 파일 기준으로 app.py import가 실패하면 sys.path에 추가
try:
    from app import User, hash_password, engine, Session
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from app import User, hash_password, engine, Session

from sqlmodel import select
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()
ADMIN_TEAM = os.getenv("ADMIN_TEAM", "__admin__")

if not ADMIN_EMAIL or not ADMIN_PASSWORD:
    print("❌ ADMIN_EMAIL / ADMIN_PASSWORD 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

if len(ADMIN_PASSWORD.encode("utf-8")) > 72:
    print("❌ bcrypt는 최대 72바이트까지만 지원합니다. 비밀번호를 줄이세요.")
    sys.exit(1)

print(f"🔑 관리자 계정 생성 중: {ADMIN_EMAIL}")

with Session(engine) as s:
    existing = s.exec(select(User).where(User.email == ADMIN_EMAIL)).first()
    hashed = hash_password(ADMIN_PASSWORD)

    if existing:
        existing.password_hash = hashed
        existing.is_admin = True
        existing.team = ADMIN_TEAM
        s.commit()
        print(f"✅ 관리자 계정 갱신 완료: {ADMIN_EMAIL}")
    else:
        u = User(
            email=ADMIN_EMAIL,
            team=ADMIN_TEAM,
            password_hash=hashed,
            is_admin=True,
        )
        s.add(u)
        s.commit()
        print(f"✅ 신규 관리자 계정 생성 완료: {ADMIN_EMAIL}")
