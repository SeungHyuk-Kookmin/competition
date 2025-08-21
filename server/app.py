from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, Session, create_engine, select
from datetime import datetime, timezone, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import io, os, logging, hashlib
from sqlalchemy import func, UniqueConstraint
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy import func, UniqueConstraint, delete

logger = logging.getLogger("uvicorn.error")

KST = timezone(timedelta(hours=9))

PRIVATE_VISIBILITY = os.getenv("PRIVATE_VISIBILITY", "hidden")  # hidden | admin | public | public_after
PRIVATE_RELEASE_AT = os.getenv("PRIVATE_RELEASE_AT", "2025-08-29-14-00-00").strip()  # 예: "2025-08-21-09-00-00" 또는 "2025-08-21 09:00:00" 또는 "2025-08-21"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ✅ 고정 KST (DST 없음)
KST = timezone(timedelta(hours=9))
TZ_NAME = "Asia/Seoul"  # 표기용
def ts_kst(dt: datetime) -> str:
    return dt.astimezone(KST).strftime("%Y-%m-%d-%H-%M-%S")

def _parse_release_at_kst(s: str):
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d-%H-%M-%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            # 날짜만 주면 00:00:00으로
            if fmt == "%Y-%m-%d":
                dt = datetime.strptime(s + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=KST)
        except Exception:
            pass
    return None

RELEASE_AT_KST = _parse_release_at_kst(PRIVATE_RELEASE_AT)

def ts_kst(dt: datetime) -> str:
    return dt.astimezone(KST).strftime("%Y-%m-%d-%H-%M-%S")

def admin_required(request: Request):  # ← 반환 타입 힌트 빼기(또는 위의 __future__ 사용)
    u = _get_current_user(request)
    if u and getattr(u, "is_admin", False):
        return u
    key = request.headers.get("X-ADMIN-KEY", "")
    if ADMIN_KEY and key == ADMIN_KEY:
        return u  # u가 None이어도 키로 통과 허용
    raise HTTPException(status_code=403, detail="Admins only.")

SUBMIT_SAVE_DIR = os.getenv("SUBMIT_SAVE_DIR", "./submissions")
os.makedirs(SUBMIT_SAVE_DIR, exist_ok=True)

# ================== 설정 ==================
TZ_NAME = os.getenv("TZ", "Asia/Seoul")
TZ = ZoneInfo(TZ_NAME)
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "10"))

ID_COL  = os.getenv("ID_COL", "ID")
Y_COL   = os.getenv("Y_COL",  "y_true")

GT_PUBLIC_PATH  = os.getenv("GT_PUBLIC_PATH",  "./data/y_test_public.csv")
GT_PRIVATE_PATH = os.getenv("GT_PRIVATE_PATH", "./data/y_test_private.csv")

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./leaderboard.db")
KEEP_BEST_PER_TEAM = True
PRED_CANDIDATES = [os.getenv("PRED_COL", "y_pred"), "pred", "prob", "probability", "score"]

# private 가시성: "hidden" | "admin" | "public"
PRIVATE_VISIBILITY = os.getenv("PRIVATE_VISIBILITY", "hidden")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")  # (옵션) 관리자 헤더 백도어

# JWT
SECRET_KEY = os.getenv("SECRET_KEY", "dev-insecure-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))

def create_access_token(data: dict, minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + timedelta(minutes=minutes)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# =========================================

app = FastAPI(title="Competition Scoring API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ================== DB 모델 ==================
class Submission(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    team: str
    public_score: Optional[float] = None
    private_score: Optional[float] = None
    rows_public: Optional[int] = None
    rows_private: Optional[int] = None
    warning: Optional[str] = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0))

class FinalPick(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    team: str
    submission_id: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    __table_args__ = (UniqueConstraint("team", "submission_id", name="uix_team_submission"),)

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str
    team: str
    password_hash: str
    is_admin: bool = False

    # ✅ 신규 필드
    name: Optional[str] = None
    student_id: Optional[str] = Field(default=None, index=True)

    __table_args__ = (
        UniqueConstraint("email", name="uix_user_email"),
        # UniqueConstraint("team", name="uix_user_team"),  # 그대로 비활성
    )

# SQLite 멀티스레드 허용
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SQLModel.metadata.create_all(engine)

def _ensure_user_columns():
    # SQLite 전용 간단 마이그레이션
    try:
        with engine.connect() as conn:
            cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info('user')").fetchall()}
            if "name" not in cols:
                conn.exec_driver_sql("ALTER TABLE user ADD COLUMN name VARCHAR;")
            if "student_id" not in cols:
                conn.exec_driver_sql("ALTER TABLE user ADD COLUMN student_id VARCHAR;")
    except Exception as e:
        logger.warning(f"User 테이블 마이그레이션 경고: {e}")

_ensure_user_columns()

# --------- Ground Truth 로드 ---------
GT_PUBLIC = None
if os.path.exists(GT_PUBLIC_PATH):
    GT_PUBLIC = pd.read_csv(GT_PUBLIC_PATH)
    if ID_COL not in GT_PUBLIC.columns or Y_COL not in GT_PUBLIC.columns:
        raise RuntimeError(f"Public GT must have [{ID_COL}, {Y_COL}]")
    GT_PUBLIC[ID_COL] = GT_PUBLIC[ID_COL].astype(str)

GT_PRIVATE = None
if os.path.exists(GT_PRIVATE_PATH):
    GT_PRIVATE = pd.read_csv(GT_PRIVATE_PATH)
    if ID_COL not in GT_PRIVATE.columns or Y_COL not in GT_PRIVATE.columns:
        raise RuntimeError(f"Private GT must have [{ID_COL}, {Y_COL}]")
    GT_PRIVATE[ID_COL] = GT_PRIVATE[ID_COL].astype(str)

if GT_PUBLIC is None and GT_PRIVATE is None:
    raise RuntimeError("At least one of public/private GT must exist.")
# -------------------------------------

GT_ALL = None
if GT_PUBLIC is not None and GT_PRIVATE is not None:
    GT_ALL = pd.concat([GT_PUBLIC, GT_PRIVATE], ignore_index=True)
    # ID 중복이 있으면 첫 라벨을 사용 (일반적으로 disjoint)
    GT_ALL = GT_ALL.drop_duplicates(subset=[ID_COL], keep="first")
elif GT_PUBLIC is not None:
    GT_ALL = GT_PUBLIC.copy()
elif GT_PRIVATE is not None:
    GT_ALL = GT_PRIVATE.copy()

class SubFile(SQLModel, table=True):
    submission_id: int = Field(primary_key=True)
    team: str
    path: str
    orig_name: Optional[str] = None

SQLModel.metadata.create_all(engine)

# === admin helper ===
def _require_admin(request: Request):
    u = _get_current_user(request)
    if u and u.is_admin:
        return u
    key = request.headers.get("X-ADMIN-KEY", "")
    if ADMIN_KEY and key == ADMIN_KEY:
        return None
    raise HTTPException(status_code=403, detail="admin only")

class DeleteReport(BaseModel):
    deleted_submissions: int = 0
    removed_final_picks: int = 0
    removed_files: int = 0
    deleted_users: int = 0

def _delete_submissions_by_ids(ids: List[int]) -> DeleteReport:
    rep = DeleteReport()
    if not ids:
        return rep
    with Session(engine) as s:
        # FinalPick 정리
        rep.removed_final_picks = s.exec(
            select(func.count()).select_from(FinalPick).where(FinalPick.submission_id.in_(ids))
        ).one()
        s.exec(delete(FinalPick).where(FinalPick.submission_id.in_(ids)))

        # 파일 삭제
        sfiles = s.exec(select(SubFile).where(SubFile.submission_id.in_(ids))).all()
        for sf in sfiles:
            try:
                if sf.path and os.path.exists(sf.path):
                    os.remove(sf.path)
                    rep.removed_files += 1
            except Exception:
                pass
        s.exec(delete(SubFile).where(SubFile.submission_id.in_(ids)))

        # 제출 삭제
        rep.deleted_submissions = s.exec(
            select(func.count()).select_from(Submission).where(Submission.id.in_(ids))
        ).one()
        s.exec(delete(Submission).where(Submission.id.in_(ids)))
        s.commit()
    return rep

# ================== 유틸/보안 ==================
def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_password(pw: str, pw_hash: str) -> bool:
    return pwd_context.verify(pw, pw_hash)

def get_user_by_email(email: str) -> Optional[User]:
    with Session(engine) as s:
        return s.exec(select(User).where(User.email == email)).first()

def _get_current_user(request: Request) -> Optional[User]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            return None
        user = get_user_by_email(email)
        return user
    except JWTError:
        return None

def _require_user(request: Request) -> User:
    user = _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

def _has_private_access(request: Request, user: Optional[User]) -> bool:
    # 1) 관리자 또는 X-ADMIN-KEY는 항상 허용
    if user and getattr(user, "is_admin", False):
        return True
    key = request.headers.get("X-ADMIN-KEY", "")
    if ADMIN_KEY and key == ADMIN_KEY:
        return True

    # 2) 가시성 정책
    if PRIVATE_VISIBILITY == "public":
        return True
    if PRIVATE_VISIBILITY == "admin":
        return False
    if PRIVATE_VISIBILITY == "hidden":
        return False
    if PRIVATE_VISIBILITY == "public_after":
        now_kst = datetime.now(timezone.utc).astimezone(KST)
        return bool(RELEASE_AT_KST and now_kst >= RELEASE_AT_KST)

    return False

# ================== 채점 유틸 ==================
def _pick_pred_column(df: pd.DataFrame) -> str:
    for c in PRED_CANDIDATES:
        if c in df.columns: return c
    numeric_cols = [c for c in df.columns if c != ID_COL and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise HTTPException(status_code=400, detail=f"No numeric prediction column found. Provide one of {PRED_CANDIDATES}")
    numeric_cols.sort(key=lambda c: float(pd.to_numeric(df[c], errors="coerce").notna().mean()), reverse=True)
    return numeric_cols[0]

def _validate_and_score(
    sub_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    *,
    require_full: bool = False,
) -> Tuple[float, int, str]:
    warn = []
    df = sub_df.copy()

    # ID 컬럼 정리
    if ID_COL not in df.columns:
        for g in ["id","Id","ID"]:
            if g in df.columns:
                df = df.rename(columns={g: ID_COL}); break
        if ID_COL not in df.columns:
            # 형식 오류 통일
            raise HTTPException(status_code=400, detail={
                "code": "BAD_SUBMISSION_FORMAT",
                "message": "제출 양식이 맞지 않습니다."
            })

    df[ID_COL] = df[ID_COL].astype(str)
    pred_col = _pick_pred_column(df)

    # 중복 ID는 즉시 에러
    if df[ID_COL].duplicated().any():
        raise HTTPException(status_code=400, detail={
            "code": "BAD_SUBMISSION_FORMAT",
            "message": "제출 양식이 맞지 않습니다."
        })

    # 숫자 변환
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")

    # ✅ 전체 매칭 강제: GT에 있는 모든 ID가 제출에 존재해야 함
    gt_ids = set(gt_df[ID_COL].astype(str))
    sub_ids = set(df[ID_COL].astype(str))
    missing_ids = gt_ids - sub_ids

    # ✅ 전체 매칭 강제: 제출된 해당 ID들의 예측값이 NaN이면 안 됨
    # (즉, GT와 겹치는 ID들 한정으로 NaN 검사)
    df_in_gt = df[df[ID_COL].isin(gt_ids)]
    nan_bad = set(df_in_gt.loc[df_in_gt[pred_col].isna(), ID_COL].astype(str))

    if require_full and (missing_ids or nan_bad):
        # 통일된 한국어 메시지
        raise HTTPException(status_code=400, detail={
            "code": "BAD_SUBMISSION_FORMAT",
            "message": "제출 양식이 맞지 않습니다."
        })

    # 기존 경고/정리 로직 유지 (require_full=False일 때만 의미)
    nan_cnt = int(df[pred_col].isna().sum())
    if nan_cnt > 0 and not require_full:
        warn.append(f"nan_pred={nan_cnt} (dropped)")
        df = df.dropna(subset=[pred_col])

    merged = gt_df.merge(df[[ID_COL, pred_col]], on=ID_COL, how="inner")
    # 전체 매칭 강제가 아니면서 겹침이 전혀 없을 때도 형식 오류로 통일
    if merged.empty:
        raise HTTPException(status_code=400, detail={
            "code": "BAD_SUBMISSION_FORMAT",
            "message": "제출 양식이 맞지 않습니다."
        })

    y_true = merged[Y_COL].values
    y_pred = merged[pred_col].values.astype(float)

    if (y_pred.min() < 0) or (y_pred.max() > 1):
        warn.append("pred_out_of_[0,1]")

    try:
        score = float(roc_auc_score(y_true, y_pred))
    except Exception:
        # 점수 계산 실패도 형식 오류로 통일
        raise HTTPException(status_code=400, detail={
            "code": "BAD_SUBMISSION_FORMAT",
            "message": "제출 양식이 맞지 않습니다."
        })

    # require_full=False일 때만 참고용 경고 남김
    if not require_full:
        missing = len(gt_ids - sub_ids)
        extra   = len(sub_ids - gt_ids)
        if missing: warn.append(f"missing_ids={missing}")
        if extra:   warn.append(f"extra_ids={extra} (ignored)")

    return score, int(len(merged)), "; ".join(warn) if warn else ""
def _today_window_utc():
    # ✅ UTC 기준 now → KST 변환 → KST 자정 ~ +1일 → 다시 UTC로
    now_kst = datetime.now(timezone.utc).astimezone(KST)
    start_kst = datetime.combine(now_kst.date(), dtime(0, 0, 0), tzinfo=KST)
    end_kst = start_kst + timedelta(days=1)
    return start_kst.astimezone(timezone.utc), end_kst.astimezone(timezone.utc)

def _best_submission_for_team(team: str, sort_field: str) -> Optional[Submission]:
    with Session(engine) as s:
        rows = s.exec(select(Submission).where(Submission.team == team)).all()
    rows = [r for r in rows if getattr(r, sort_field) is not None]
    if not rows: return None
    rows.sort(key=lambda r: (-getattr(r, sort_field), r.received_at))
    return rows[0]

def _best_of_two(submissions: List[Submission], sort_field: str) -> Optional[Submission]:
    cands = [s for s in submissions if s is not None and getattr(s, sort_field) is not None]
    if not cands: return None
    cands.sort(key=lambda r: (-getattr(r, sort_field), r.received_at))
    return cands[0]

# ================== 스키마 ==================
class LeaderboardItem(BaseModel):
    team: str
    public_score: Optional[float] = None
    private_score: Optional[float] = None
    rows_public: Optional[int] = None
    rows_private: Optional[int] = None
    received_at: str

class SubmitResponse(LeaderboardItem):
    submission_id: int
    used_today: Optional[int] = None
    remaining_today: Optional[int] = None
    daily_limit: Optional[int] = None

class FinalizeBody(BaseModel):
    submission_ids: List[int]  # 1~2개

class RegisterBody(BaseModel):
    email: str
    team: str
    password: str
    name: str          # ✅ 추가
    student_id: str    # ✅ 추가 (8자리 숫자)

class LoginBody(BaseModel):
    email: str
    password: str

# ================== 인증 엔드포인트 ==================
@app.post("/auth/register")
def register(body: RegisterBody):
    body.email = body.email.strip().lower()
    body.team = body.team.strip()
    body.name = body.name.strip()
    body.student_id = body.student_id.strip()

    if not body.email or not body.team or not body.password or not body.name or not body.student_id:
        raise HTTPException(status_code=400, detail={"code":"REQUIRED_FIELDS","message":"아이디/팀/비밀번호/이름/학번은 필수입니다."})

    # 학번 형식 검증: 숫자 8자리
    if not (body.student_id.isdigit() and len(body.student_id) == 8):
        raise HTTPException(status_code=422, detail={"code":"INVALID_STUDENT_ID","message":"학번은 숫자 8자리여야 합니다."})

    with Session(engine) as s:
        if s.exec(select(User).where(User.email == body.email)).first():
            raise HTTPException(status_code=409, detail={"code":"EMAIL_EXISTS","message":"이미 등록된 아이디입니다."})
        # 선택: 학번 중복 방지
        if s.exec(select(User).where(User.student_id == body.student_id)).first():
            raise HTTPException(status_code=409, detail={"code":"STUDENT_ID_EXISTS","message":"이미 등록된 학번입니다."})

        user = User(
            email=body.email,
            team=body.team,
            password_hash=hash_password(body.password),
            is_admin=False,
            name=body.name,
            student_id=body.student_id
        )
        s.add(user); s.commit()
    return {"ok": True}

@app.post("/auth/login")
def login(body: LoginBody):
    email = body.email.strip().lower()
    user = get_user_by_email(email)
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(
    status_code=401,
    detail={"code": "INVALID_CREDENTIALS", "message": "아이디 또는 비밀번호가 올바르지 않습니다."},
    headers={"WWW-Authenticate": "Bearer"}  # 토큰 기반이면 권장
)
    token = create_access_token({"sub": user.email, "team": user.team, "is_admin": bool(user.is_admin)})
    return {"access_token": token, "token_type": "bearer", "team": user.team, "is_admin": bool(user.is_admin)}

@app.get("/auth/me")
def me(request: Request):
    user = _require_user(request)
    return {"email": user.email, "team": user.team, "is_admin": user.is_admin,
            "name": user.name, "student_id": user.student_id}

# ================== 제출/조회/최종 ==================
@app.post("/submit", response_model=SubmitResponse)
async def submit(request: Request, file: UploadFile = File(...)):
    try:
        user = _require_user(request)
        team = user.team

        # 일일 제출 제한
        start_utc, end_utc = _today_window_utc()
        with Session(engine) as s:
            cnt = len(
                s.exec(
                    select(Submission)
                    .where(Submission.team == team)
                    .where(Submission.received_at >= start_utc)
                    .where(Submission.received_at < end_utc)
                ).all()
            )
        if cnt >= DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "DAILY_LIMIT_EXCEEDED",
                    "message": f"하루 제출 한도({DAILY_LIMIT})를 초과했습니다.",
                    "team": team,
                    "reset_at_local": f"00:00 {TZ_NAME}",
                },
                # headers={"Retry-After": ...}
            )

        # CSV 읽기
        content = await file.read()
        sub_df = pd.read_csv(io.BytesIO(content))

        # 채점
        warnings = []
        public_score = rows_pub = None
        private_score = rows_pri = None
        
        if GT_PUBLIC is not None:
            ps, rp, w = _validate_and_score(sub_df, GT_PUBLIC, require_full=True)   # ✅ 전체 매칭 강제
            public_score, rows_pub = ps, rp
            if w:
                warnings.append(f"[public] {w}")
        
        if GT_PRIVATE is not None:
            ps, rp, w = _validate_and_score(sub_df, GT_PRIVATE, require_full=True)  # ✅ 전체 매칭 강제
            private_score, rows_pri = ps, rp
            if w:
                warnings.append(f"[private] {w}")

        warn_str = " | ".join(warnings) if warnings else None

        # DB 저장
        with Session(engine) as s:
            rec = Submission(
                team=team,
                public_score=public_score,
                private_score=private_score,
                rows_public=rows_pub,
                rows_private=rows_pri,
                warning=warn_str,
            )
            s.add(rec)
            s.commit()
            s.refresh(rec)

        # 원본 CSV 파일 보관 (최종 전체 테스트 재채점용)
        try:
            safe_team = "".join([c if c.isalnum() or c in "-_." else "_" for c in team])
            fname = f"{rec.id}_{safe_team}.csv"
            fpath = os.path.join(SUBMIT_SAVE_DIR, fname)
            with open(fpath, "wb") as fw:
                fw.write(content)
            with Session(engine) as s2:
                s2.add(SubFile(submission_id=rec.id, team=team, path=fpath, orig_name=getattr(file, "filename", None)))
                s2.commit()
        except Exception as e:
            logger.warning("Failed to persist submission file: %s", e)

        # 응답 (비공개 가시성 정책 반영)
        can_private = _has_private_access(request, user)
        used_today = cnt + 1
        remaining_today = max(0, DAILY_LIMIT - used_today)
        return SubmitResponse(
            submission_id=rec.id,
            team=team,
            public_score=public_score,
            private_score=private_score if can_private else None,
            rows_public=rows_pub,
            rows_private=rows_pri if can_private else None,
            received_at=ts_kst(rec.received_at),
            used_today=used_today,
            remaining_today=remaining_today,
            daily_limit=DAILY_LIMIT,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Submit failed: %s", e)
        raise HTTPException(status_code=500, detail=f"internal error: {e}")

@app.get("/my_score")
def my_score(request: Request):
    """ 로그인 사용자 본인 점수/히스토리/최종 후보 조회 """
    user = _require_user(request)
    team = user.team

    best_pub = _best_submission_for_team(team, "public_score")
    best_pri = _best_submission_for_team(team, "private_score")

    with Session(engine) as s:
        history = s.exec(
            select(Submission).where(Submission.team == team).order_by(Submission.received_at.desc()).limit(50)
        ).all()
        picks = s.exec(
            select(FinalPick).where(FinalPick.team == team).order_by(FinalPick.created_at.asc())
        ).all()
        pick_subs = [s.get(Submission, p.submission_id) for p in picks]

    def _row(sub: Optional[Submission]):
        if not sub: return None
        return {"submission_id": sub.id, "public_score": sub.public_score,
                "private_score": sub.private_score, "received_at": ts_kst(sub.received_at)}

    return {
        "team": team,
        "best_public": _row(best_pub),
        "best_private": _row(best_pri),
        "final_picks": [_row(s) for s in pick_subs if s],
        "history": [_row(s) for s in history],
    }

class FinalizeBody(BaseModel):
    submission_ids: List[int]

@app.post("/finalize")
def finalize(request: Request, body: FinalizeBody):
    user = _require_user(request)
    team = user.team
    ids = list(dict.fromkeys(body.submission_ids or []))
    if not (1 <= len(ids) <= 2):
        raise HTTPException(status_code=400, detail="Provide 1 or 2 submission_ids")

    with Session(engine) as s:
        subs = []
        for sid in ids:
            sub = s.get(Submission, sid)
            if not sub:
                raise HTTPException(status_code=404, detail=f"Submission {sid} not found")
            if sub.team != team:
                raise HTTPException(status_code=403, detail=f"Submission {sid} not owned by team {team}")
            subs.append(sub)

        # ✅ 기존 선택 삭제 (SQLAlchemy 2.x 안전 방식)
        s.exec(delete(FinalPick).where(FinalPick.team == team))
        s.commit()

        # 새 선택 저장
        for sid in ids:
            s.add(FinalPick(team=team, submission_id=sid))
        s.commit()

        best_pub = _best_of_two(subs, "public_score")
        best_pri = _best_of_two(subs, "private_score")
        return {
            "team": team,
            "selected_ids": ids,
            "best_public_among_selected": {"submission_id": best_pub.id if best_pub else None,
                                           "score": best_pub.public_score if best_pub else None},
            "best_private_among_selected": {"submission_id": best_pri.id if best_pri else None,
                                            "score": best_pri.private_score if best_pri else None}
        }

@app.get("/my_quota")
def my_quota(request: Request):
    user = _require_user(request)
    start_utc, end_utc = _today_window_utc()
    with Session(engine) as s:
        used = s.exec(
            select(func.count()).select_from(Submission)
            .where(Submission.team == user.team)
            .where(Submission.received_at >= start_utc)
            .where(Submission.received_at < end_utc)
        ).one()
    used = int(used)
    remaining = max(0, DAILY_LIMIT - used)
    return {
        "team": user.team,
        "used_today": used,
        "remaining": remaining,
        "daily_limit": DAILY_LIMIT,
        "reset_at_local": f"00:00 {TZ_NAME}",
    }

# 관리자: 유저 목록 조회 (옵션: 팀 필터)
@app.get("/admin/users")
def admin_list_users(request: Request, _: User = Depends(admin_required), team: Optional[str] = None, limit: int = 500):
    with Session(engine) as s:
        q = select(User)
        if team:
            q = q.where(User.team == team)
        q = q.order_by(User.id.desc()).limit(limit)
        rows = s.exec(q).all()
    return [{"email": u.email, "team": u.team, "is_admin": bool(u.is_admin),
             "name": u.name, "student_id": u.student_id} for u in rows]

# 최근 제출 목록(관리자)
@app.get("/admin/submissions")
def admin_list_submissions(request: Request, team: Optional[str] = None, limit: int = 200):
    _require_admin(request)
    with Session(engine) as s:
        q = select(Submission)
        if team:
            q = q.where(Submission.team == team)
        q = q.order_by(Submission.received_at.desc()).limit(limit)
        rows = s.exec(q).all()

        # 원본 CSV 경로 미리 모아두기
        ids = [r.id for r in rows]
        sfiles = s.exec(select(SubFile).where(SubFile.submission_id.in_(ids))).all()

    file_map = {sf.submission_id: sf for sf in sfiles}

    out = []
    for r in rows:
        # ✅ 최종 리더보드(Private)와 동일한 스코어 산출 로직:
        # 1) GT_ALL로 재채점 성공 시 그 값을 사용
        # 2) 실패하거나 파일 없으면 기존 저장된 private_score 폴백
        final_like_private = None
        sf = file_map.get(r.id)
        if GT_ALL is not None and sf and sf.path and os.path.exists(sf.path):
            sc, _rows, _w = _score_file_on_gt(sf.path, GT_ALL)
            if sc is not None:
                final_like_private = float(sc)
        if final_like_private is None and r.private_score is not None:
            final_like_private = float(r.private_score)

        item = {
            "id": r.id,
            "team": r.team,
            "received_at": ts_kst(r.received_at),
            "public_score": r.public_score,
            # ⬇️ 여기의 private_score가 이제 ‘최종 리더보드(Private)’와 동일 로직
            "private_score": round(final_like_private, 6) if final_like_private is not None else None,
            # 참고용으로 원래 저장치도 함께 내려줌(원치 않으면 이 줄 삭제)
            # "private_score_raw": round(float(r.private_score), 6) if r.private_score is not None else None,
            "rows_public": r.rows_public,
            "rows_private": r.rows_private,
            "warning": r.warning,
        }
        out.append(item)

    return out


# 개별 제출 삭제
@app.delete("/admin/submission/{submission_id}", response_model=DeleteReport)
def admin_delete_submission(submission_id: int, request: Request):
    _require_admin(request)
    return _delete_submissions_by_ids([submission_id])

# 팀의 모든 제출 삭제
@app.delete("/admin/team/{team}/submissions", response_model=DeleteReport)
def admin_delete_team_submissions(team: str, request: Request):
    _require_admin(request)
    with Session(engine) as s:
        ids = [r[0] for r in s.exec(select(Submission.id).where(Submission.team == team)).all()]
    return _delete_submissions_by_ids(ids)

# 사용자(계정) 삭제 (옵션: 같은 팀 제출도 함께 삭제)
@app.delete("/admin/user/{email}", response_model=DeleteReport)
def admin_delete_user(email: str, request: Request, cascade_team_submissions: bool = False):
    _require_admin(request)
    email = email.strip().lower()
    rep = DeleteReport()
    with Session(engine) as s:
        u = s.exec(select(User).where(User.email == email)).first()
        if not u:
            raise HTTPException(status_code=404, detail="user not found")
        team = u.team
        s.exec(delete(User).where(User.email == email))
        rep.deleted_users = 1
        s.commit()
    if cascade_team_submissions:
        team_rep = admin_delete_team_submissions(team, request)
        rep.deleted_submissions += team_rep.deleted_submissions
        rep.removed_final_picks += team_rep.removed_final_picks
        rep.removed_files += team_rep.removed_files
    return rep

# ---- 일반 리더보드 (팀별 최고 1개) ----
def _best_per_team(sort_key: str):
    with Session(engine) as s:
        rows = s.exec(select(Submission)).all()
    best = {}
    for r in rows:
        score = getattr(r, sort_key)
        if score is None: continue
        if (r.team not in best) or (score > getattr(best[r.team], sort_key)) or \
           (score == getattr(best[r.team], sort_key) and r.received_at < best[r.team].received_at):
            best[r.team] = r
    return sorted(best.values(), key=lambda x: (-getattr(x, sort_key), x.received_at))

@app.get("/leaderboard/public")
def leaderboard_public(limit: int = 100):
    items = _best_per_team("public_score")[:limit]
    return [{"team": r.team,
             "public_score": round(float(r.public_score), 6) if r.public_score is not None else None,
             "rows_public": r.rows_public,
             "received_at": ts_kst(r.received_at),  # ✅
             "warning": r.warning} for r in items]

@app.get("/leaderboard/private")
def leaderboard_private(request: Request, limit: int = 100):
    user = _get_current_user(request)
    if not _has_private_access(request, user):
        raise HTTPException(status_code=403, detail="Forbidden.")
    items = _best_per_team("private_score")[:limit]
    return [{"team": r.team,
             "private_score": round(float(r.private_score), 6) if r.private_score is not None else None,
             "rows_private": r.rows_private,
             "received_at": ts_kst(r.received_at),  # ✅
             "warning": r.warning} for r in items]

def _csv_response(name: str, df: pd.DataFrame) -> StreamingResponse:
    buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})

@app.get("/leaderboard/public_csv")
def leaderboard_public_csv(limit: int = 100):
    items = _best_per_team("public_score")[:limit]
    if not items:
        return _csv_response("leaderboard_public", pd.DataFrame(columns=["team","public_score","rows_public","received_at"]))
    df = pd.DataFrame([{"team": r.team,
                        "public_score": round(float(r.public_score), 6) if r.public_score is not None else None,
                        "rows_public": r.rows_public,
                        "received_at": ts_kst(r.received_at)} for r in items])
    return _csv_response("leaderboard_public", df)

@app.get("/leaderboard/private_csv")
def leaderboard_private_csv(request: Request, limit: int = 100):
    user = _get_current_user(request)
    if not _has_private_access(request, user):
        raise HTTPException(status_code=403, detail="Forbidden.")
    items = _best_per_team("private_score")[:limit]
    if not items:
        return _csv_response("leaderboard_private",
                             pd.DataFrame(columns=["team","private_score","rows_private","received_at"]))
    df = pd.DataFrame([{"team": r.team,
                        "private_score": round(float(r.private_score), 6) if r.private_score is not None else None,
                        "rows_private": r.rows_private,
                        "received_at": ts_kst(r.received_at)} for r in items])
    return _csv_response("leaderboard_private", df)

# ---- 최종 리더보드 (팀별 선정 1~2개 중 최고) ----
def _final_best_map(sort_field: str):
    with Session(engine) as s:
        picks = s.exec(select(FinalPick)).all()
        if not picks: return {}
        team_to_ids = {}
        for p in picks:
            team_to_ids.setdefault(p.team, set()).add(p.submission_id)
        ids = {p.submission_id for p in picks}
        subs = s.exec(select(Submission).where(Submission.id.in_(list(ids)))).all()
    result = {}
    for team, idset in team_to_ids.items():
        cands = [sub for sub in subs if sub.id in idset and getattr(sub, sort_field) is not None]
        if not cands: continue
        cands.sort(key=lambda r: (-getattr(r, sort_field), r.received_at))
        result[team] = cands[0]
    return result

@app.get("/final/leaderboard_public")
def final_leaderboard_public(limit: int = 100):
    best_map = _final_best_map("public_score")
    items = sorted(best_map.values(), key=lambda r: (-r.public_score, r.received_at))[:limit]
    return [{
        "team": r.team,
        "submission_id": r.id,
        "public_score": round(float(r.public_score), 6) if r.public_score is not None else None,
        "rows_public": r.rows_public,                     # ✅ 추가
        "received_at": ts_kst(r.received_at)
    } for r in items]

@app.get("/final/leaderboard_private")
def final_leaderboard_private(request: Request, limit: int = 100):
    user = _get_current_user(request)
    if not _has_private_access(request, user):
        raise HTTPException(status_code=403, detail="Forbidden.")

    # 1) 선택된 후보 수집
    with Session(engine) as s:
        picks = s.exec(select(FinalPick)).all()
        if not picks:
            return []
        team_to_ids = {}
        for p in picks:
            team_to_ids.setdefault(p.team, set()).add(p.submission_id)

        ids = {pid for v in team_to_ids.values() for pid in v}
        subs_list = s.exec(select(Submission).where(Submission.id.in_(list(ids)))).all()
        files_list = s.exec(select(SubFile).where(SubFile.submission_id.in_(list(ids)))).all()

    subs  = {r.id: r for r in subs_list}
    files = {r.submission_id: r for r in files_list}

    def _rescore_like_final(sub: Submission, sf: Optional[SubFile]):
        """GT_ALL 재채점 성공 시 그 점수 사용, 실패 시 기존 private_score/rows_private 폴백."""
        try:
            if GT_ALL is not None and sf and sf.path and os.path.exists(sf.path):
                sc, rows, _w = _score_file_on_gt(sf.path, GT_ALL)
                if sc is not None:
                    return float(sc), rows
        except Exception as e:
            logger.warning("final/private rescore failed (id=%s): %s", getattr(sub, "id", None), e)
        if sub and sub.private_score is not None:
            return float(sub.private_score), (sub.rows_private if sub.rows_private is not None else None)
        return None, None

    items = []
    for team, idset in team_to_ids.items():
        best = None  # dict로 보관
        for sid in idset:
            sub = subs.get(sid)
            if not sub:
                continue
            sf  = files.get(sid)
            score, rows_used = _rescore_like_final(sub, sf)
            if score is None:
                continue

            # 정렬/타이브레이크용 키 (UTC epoch; 안전하게 숫자)
            recv_dt = sub.received_at or datetime.now(timezone.utc)
            recv_key = recv_dt.timestamp()

            cand = {
                "team": team,
                "submission_id": sid,
                "public_score": round(float(sub.public_score), 6) if sub.public_score is not None else None,
                "private_score": round(float(score), 6),
                "rows": rows_used,
                "received_at": ts_kst(sub.received_at) if sub and sub.received_at else None,
                "_recv_key": recv_key,  # 내부 정렬 전용
            }

            if (best is None) or \
               (cand["private_score"] > best["private_score"]) or \
               (cand["private_score"] == best["private_score"] and cand["_recv_key"] < best["_recv_key"]):
                best = cand

        if best:
            items.append(best)

    # 안정 정렬: 점수 내림차순 → 제출시각 오름차순
    items.sort(key=lambda r: (-r["private_score"], r["_recv_key"]))
    # 내부 키 제거
    for it in items:
        it.pop("_recv_key", None)

    return items[:limit]

@app.get("/final/leaderboard_public_csv")
def final_leaderboard_public_csv(limit: int = 100):
    best_map = _final_best_map("public_score")
    items = sorted(best_map.values(), key=lambda r: (-r.public_score, r.received_at))[:limit]
    if not items:
        return _csv_response("final_leaderboard_public", pd.DataFrame(columns=["team","submission_id","public_score","received_at"]))
    df = pd.DataFrame([{"team": r.team, "submission_id": r.id,
                        "public_score": round(float(r.public_score), 6) if r.public_score is not None else None,
                        "received_at": ts_kst(r.received_at)} for r in items])
    return _csv_response("final_leaderboard_public", df)

@app.get("/final/leaderboard_private_csv")
def final_leaderboard_private_csv(request: Request, limit: int = 100):
    data = final_leaderboard_private(request, limit=limit)
    if not data:
        return _csv_response("final_leaderboard_private",
                             pd.DataFrame(columns=["team","submission_id","private_score","received_at"]))
    df = pd.DataFrame(data)
    return _csv_response("final_leaderboard_private", df)

@app.get("/health")
def health():
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    now_kst = now_utc.astimezone(KST)
    return {
        "ok": True,
        "tz": "Asia/Seoul",
        "daily_limit": DAILY_LIMIT,
        "has_public": GT_PUBLIC is not None,
        "has_private": GT_PRIVATE is not None,
        "private_visibility": PRIVATE_VISIBILITY,
        "auth": "jwt",
        "now_utc": now_utc.isoformat(),
        "now_kst": ts_kst(now_utc),
        "private_release_at_kst": RELEASE_AT_KST.strftime("%Y-%m-%d-%H-%M-%S") if RELEASE_AT_KST else None,
        "private_released": bool(RELEASE_AT_KST and now_kst >= RELEASE_AT_KST),
    }
