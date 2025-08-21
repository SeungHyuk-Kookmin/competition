import os
import requests
import pandas as pd
import streamlit as st
from sqlalchemy import func, UniqueConstraint, delete
from datetime import datetime
import pytz
from datetime import datetime, timezone, timedelta, time as dtime

API = os.getenv("API_URL", "http://localhost:8000")

info = {}
try:
    info = requests.get(f"{API}/health", timeout=10).json()
except Exception:
    pass

private_visibility = info.get("private_visibility", "hidden")
private_release_at = info.get("private_release_at_kst")
private_released = info.get("private_released", False)

st.set_page_config(page_title="ML STUDY Competition", layout="wide")
st.title("📈 D&A X WEAVE 여름방학 ML STUDY Competition")
st.caption("일일 제출 제한: 팀당 10회 (자정 기준)")

def parse_err(resp):
    try:
        js = resp.json()
    except Exception:
        return resp.text
    # FastAPI 표준(detail), 커스텀(code/message) 모두 처리
    if isinstance(js, dict):
        if "message" in js:
            return js.get("message")
        d = js.get("detail")
        if isinstance(d, dict):
            return d.get("message") or d.get("detail") or str(d)
        if isinstance(d, str):
            return d
    return str(js)

# -------- Session State --------
if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.team = None
    st.session_state.is_admin = False

def authed_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

# -------- Sidebar: Auth --------

def fetch_quota():
    if not st.session_state.token:
        return None
    try:
        r = requests.get(f"{API}/my_quota", headers=authed_headers(), timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

with st.sidebar:
    st.subheader("로그인")
    if not st.session_state.token:
        tab_login, tab_register = st.tabs(["로그인", "회원가입"])
        with tab_login:
            email = st.text_input("아이디", key="login_email")
            pw = st.text_input("패스워드", type="password", key="login_pw")
            if st.button("로그인"):
                try:
                    r = requests.post(f"{API}/auth/login",
                                      json={"email": email.strip().lower(), "password": pw},
                                      timeout=30)
                    if r.status_code == 200:
                        js = r.json()
                        st.session_state.token = js["access_token"]
                        st.session_state.team = js["team"]
                        st.session_state.is_admin = js.get("is_admin", False)
                        st.success(f"환영합니다, {st.session_state.team} 팀!")
                        st.rerun()
                    else:
                        st.error(parse_err(r))
                except Exception as e:
                    st.error(str(e))
        with tab_register:
            email_r = st.text_input("아이디", key="reg_email")
            team_r = st.text_input("팀명", key="reg_team")
            name_r = st.text_input("이름", key="reg_name")                 # ✅
            sid_r  = st.text_input("학번(숫자 8자리)", key="reg_sid")       # ✅
            pw_r = st.text_input("패스워드", type="password", key="reg_pw")
            if st.button("회원가입"):
                # 클라 단 검증(선택)
                if not (sid_r.isdigit() and len(sid_r) == 8):
                    st.error("학번은 숫자 8자리여야 합니다.")
                else:
                    try:
                        r = requests.post(
                            f"{API}/auth/register",
                            json={
                                "email": email_r.strip().lower(),
                                "team": team_r.strip(),
                                "password": pw_r,
                                "name": name_r.strip(),
                                "student_id": sid_r.strip(),
                            },
                            timeout=30
                        )
                        if r.status_code == 200:
                            st.success("가입 완료! 로그인 해주세요.")
                        else:
                            st.error(parse_err(r))
                    except Exception as e:
                        st.error(str(e))
                        
    else:
        st.write(f"**팀:** {st.session_state.team}")
        st.write(f"역할: {'관리자' if st.session_state.is_admin else '참가자'}")
        q = fetch_quota()
        if q:
            st.info(f"오늘 남은 제출: {q['remaining']}/{q['daily_limit']}")
        if st.button("로그아웃"):
            st.session_state.token = None
            st.session_state.team = None
            st.session_state.is_admin = False
            st.rerun()



# -------- Helpers --------
def show_board(endpoint_json, endpoint_csv, cols_order, headers=None, date_only=True, labels=None):
    c1, c2 = st.columns([3, 1])
    with c1:
        try:
            r = requests.get(f"{API}{endpoint_json}", headers=headers, timeout=30)
            if r.status_code == 200:
                df = pd.DataFrame(r.json())
                if not df.empty:
                    if date_only and "received_at" in df.columns:
                        df["date"] = df["received_at"].astype(str).str[:10]

                    cols = [("date" if c == "received_at" else c) for c in cols_order]

                    rename = {
                        "team": "팀",
                        "submission_id": "ID",
                        "public_score": "Public Score",
                        "private_score": "Private Score",
                        "rows_public": "제출수",
                        "rows_private": "제출수",
                        "rows": "제출수",
                        "date": "등록일",   # ✅ 기본값을 '등록일'로
                    }
                    if labels:
                        rename.update(labels)

                    present = [c for c in cols if c in df.columns]
                    disp = df.rename(columns=rename)
                    disp_cols = [rename.get(c, c) for c in present]

                    st.dataframe(disp[disp_cols], use_container_width=True)
                else:
                    st.info("데이터가 없습니다.")
            else:
                st.error(f"{r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"요청 실패: {e}")

# -------- Tabs --------
def add_tab(tabs_list, name):
    if name not in tabs_list:
        tabs_list.append(name)

# private 탭 노출 조건
show_private_tab = (
    (private_visibility in ("public", "public_after")) or
    (private_visibility == "admin" and st.session_state.is_admin)
)
show_final_tabs = st.session_state.is_admin

tabs = ["Upload Submission", "Score(Public)", "리더보드(Public)"]
if show_private_tab:
    add_tab(tabs, "리더보드(Private)")
if show_final_tabs:
    add_tab(tabs, "최종 리더보드(Public)")
    add_tab(tabs, "최종 리더보드(Private)")

if st.session_state.is_admin:
    add_tab(tabs, "관리자 도구")

tab_objs = st.tabs(tabs)
tab_idx = 0

# --- 제출 업로드 ---
with tab_objs[tab_idx]:
    st.subheader("Upload Submission")
    if not st.session_state.token:
        st.warning("로그인 후 이용해주세요.")
    else:
        # 상단에 현재 쿼터 노출
        q = fetch_quota()
        if q:
            st.caption(f"오늘 남은 제출: {q['remaining']}/{q['daily_limit']} (리셋: {q['reset_at_local']})")

        sub = st.file_uploader("submission.csv (ID + y_pred)", type=["csv"])
        if st.button("제출"):
            if not sub:
                st.error("CSV를 업로드해주세요.")
            else:
                try:
                    files = {"file": (sub.name, sub.getvalue(), "text/csv")}
                    r = requests.post(f"{API}/submit", files=files, headers=authed_headers(), timeout=60)
                    if r.status_code == 200:
                        js = r.json()
                        st.success(
                            f"제출 완료! Public: {js.get('public_score')}  / "
                            f"Private: {'—' if js.get('private_score') is None else js.get('private_score')}"
                        )
                        # ✅ 응답에 포함된 최신 쿼터로 갱신 표시
                        if js.get("daily_limit") is not None:
                            st.info(f"오늘 남은 제출: {js.get('remaining_today')}/{js.get('daily_limit')}")
                    else:
                        # 한도 초과(429) 등 한국어 메시지 표시
                        st.error(parse_err(r))
                except Exception as e:
                    st.error(str(e))
tab_idx += 1

# --- 내 점수(공개만) ---
with tab_objs[tab_idx]:
    st.subheader("Score (Public) / History / 최종 후보")
    if not st.session_state.token:
        st.warning("로그인 후 이용해주세요.")
    else:
        r = requests.get(f"{API}/my_score", headers=authed_headers(), timeout=30)
        if r.status_code != 200:
            st.error(f"{r.status_code} - {r.text}")
        else:
            js = r.json()

            # 상단 베스트 (Public만)
            best_pub = js.get("best_public") or {}
            bst_date = (best_pub.get("received_at") or "")[:10]  # YYYY-MM-DD만
            st.metric("Best Public Score", value=f"{best_pub.get('public_score'):.6f}" if best_pub.get("public_score") is not None else "—")
            st.caption(f"submission_id: {best_pub.get('submission_id')}, date {bst_date}")

            st.divider()

            # 히스토리: public만 보이게 + 한국시간 포맷
            hist = pd.DataFrame(js.get("history") or [])
            if not hist.empty:
                # 날짜 열 생성 (문자열 앞 10자리만 사용)
                hist["date_only"] = hist["received_at"].astype(str).str[:10]

                # 날짜만 보이는 뷰
                view = hist[["submission_id", "date_only", "public_score"]].copy()
                view = view.rename(columns={
                    "submission_id": "ID",
                    "date_only": "제출일",
                    "public_score": "Public Score"
                })

                st.write("#### Submission History")
                st.dataframe(view, use_container_width=True)

                st.write("#### 최종 후보 선택 (최대 2개)")
                pick_df = view.copy()
                pick_df["선택"] = False

                edited = st.data_editor(
                    pick_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "선택": st.column_config.CheckboxColumn("선택", help="최대 2개 선택", default=False),
                        "ID": st.column_config.NumberColumn("ID", disabled=True),
                        "제출일": st.column_config.TextColumn("제출일", disabled=True),  # ⬅ 날짜만
                        "Public Score": st.column_config.NumberColumn("Public Score", disabled=True, format="%.6f"),
                    },
                    disabled=["ID","제출일","Public Score"],
                )

                if st.button("최종 후보 등록"):
                    picked_ids = edited.loc[edited["선택"], "ID"].astype(int).tolist()
                    if len(picked_ids) == 0:
                        st.error("최소 1개 이상 선택해주세요.")
                    elif len(picked_ids) > 2:
                        st.error("최대 2개까지만 선택할 수 있어요.")
                    else:
                        r2 = requests.post(f"{API}/finalize", headers=authed_headers(),
                                        json={"submission_ids": picked_ids}, timeout=30)
                        if r2.status_code == 200:
                            st.success(f"최종 후보 등록 완료! 선택: {picked_ids}")
                        else:
                            st.error(f"{r2.status_code} - {r2.text}")
            else:
                st.info("아직 제출 기록이 없습니다.")
tab_idx += 1

# --- 리더보드(공개) ---
with tab_objs[tab_idx]:
    st.subheader("리더보드 (Public)")
    show_board(
        "/leaderboard/public", "/leaderboard/public_csv",
        ["team", "public_score", "rows_public", "received_at"],
        labels={
            "public_score": "점수",
            "rows_public": "제출수",
            "date": "등록일",
        }
    )
tab_idx += 1

# --- 리더보드(비공개) ---
if show_private_tab:
    with tab_objs[tab_idx]:
        st.subheader("리더보드 (Private)")
        if private_visibility == "public_after" and not private_released:
            when = private_release_at or "(미설정)"
            st.info(f"비공개 리더보드는 아직 비공개 상태입니다. 공개 예정: {when} (KST)")
        else:
            headers = authed_headers() if private_visibility == "admin" else None
            show_board("/leaderboard/private", "/leaderboard/private_csv",
                       ["team","private_score","rows_private","received_at"], headers=headers)
    tab_idx += 1

# --- 최종 리더보드 (관리자 전용) ---
if show_final_tabs:
    with tab_objs[tab_idx]:
        st.subheader("최종 리더보드 (Public, best-of-two)")
        show_board(
            "/final/leaderboard_public", "/final/leaderboard_public_csv",
            ["team", "public_score", "rows_public", "received_at"],
            headers=authed_headers(),
            labels={
                "public_score": "점수",
                "rows_public": "제출수",
                "date": "등록일",
            }
        )
    tab_idx += 1

    with tab_objs[tab_idx]:
        st.subheader("최종 리더보드 (Private, best-of-two)")
        show_board(
            "/final/leaderboard_private", "/final/leaderboard_private_csv",
            ["team", "public_score", "private_score", "rows", "received_at"],
            headers=authed_headers(),
            labels={
                "public_score": "Public",
                "private_score": "Private",
                "rows": "제출수",
                "date": "등록일",
            }
        )
    tab_idx += 1

# --- 관리자 도구 ---
if st.session_state.is_admin:
    with tab_objs[-1]:
        st.subheader("관리자 도구")

        # 섹션 1: 최근 제출 조회 & 선택 삭제
        st.markdown("#### 최근 제출 조회/삭제")
        colf1, colf2 = st.columns([2, 1])
        with colf1:
            team_filter = st.text_input("팀 필터(옵션)", key="admin_team_filter")
        with colf2:
            limit = st.number_input("표시 개수", min_value=10, max_value=1000, value=50, step=10, key="admin_limit")

        if st.button("목록 불러오기", key="btn_list_subs"):
            try:
                params = {"limit": limit}
                if team_filter.strip():
                    params["team"] = team_filter.strip()
                r = requests.get(f"{API}/admin/submissions", params=params, headers=authed_headers(), timeout=30)
                if r.status_code == 200:
                    st.session_state["_admin_subs"] = pd.DataFrame(r.json())
                else:
                    st.error(parse_err(r))
            except Exception as e:
                st.error(str(e))

        df_subs = st.session_state.get("_admin_subs")
        if isinstance(df_subs, pd.DataFrame) and not df_subs.empty:
            # 필요한 컬럼만 추림 (백엔드에서 private_score는 이미 최종 리더보드(Private)와 동일 로직)
            cols_needed = ["team", "id", "public_score", "private_score", "received_at"]
            present = [c for c in cols_needed if c in df_subs.columns]
            view = df_subs[present].copy()
        
            # 타입 정리 + 표시명 변경
            if "id" in view.columns:
                view["id"] = pd.to_numeric(view["id"], errors="coerce").astype("Int64")
            rename_map = {
                "team": "팀",
                "id": "ID",
                "public_score": "Public Score",
                "private_score": "Private Score",
                "received_at": "제출일시",
            }
            view = view.rename(columns=rename_map)
        
            # 체크박스 컬럼 추가
            view["삭제"] = False
        
            # 컬럼 순서 강제
            order = ["팀", "ID", "Public Score", "Private Score", "제출일시", "삭제"]
            order = [c for c in order if c in view.columns]
            view = view[order]
        
            edited = st.data_editor(
                view,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "팀": st.column_config.TextColumn("팀", disabled=True),
                    "ID": st.column_config.NumberColumn("ID", disabled=True),
                    "Public Score": st.column_config.NumberColumn("Public Score", disabled=True, format="%.6f"),
                    "Private Score": st.column_config.NumberColumn("Private Score", disabled=True, format="%.6f"),
                    "제출일시": st.column_config.TextColumn("제출일시", disabled=True),
                    "삭제": st.column_config.CheckboxColumn("삭제", help="삭제할 제출 선택", default=False),
                },
                disabled=["팀", "ID", "Public Score", "Private Score", "제출일시"],
            )
        
            if st.button("선택 제출 삭제", key="btn_delete_selected_clean"):
                ids = edited.loc[edited["삭제"], "ID"]
                ids = ids.dropna().astype(int).tolist() if "ID" in edited.columns else []
                if not ids:
                    st.warning("선택된 제출이 없습니다.")
                else:
                    ok, fail = 0, 0
                    for sid in ids:
                        rr = requests.delete(f"{API}/admin/submission/{sid}", headers=authed_headers(), timeout=30)
                        if rr.status_code == 200:
                            ok += 1
                        else:
                            fail += 1
                    st.success(f"삭제 완료: {ok}건, 실패: {fail}건")
                    st.session_state.pop("_admin_subs", None)
                    st.rerun()
        else:
            st.info("최근 제출이 없습니다.")
        st.divider()

        # 섹션 2: 팀 제출 전체 삭제
        st.markdown("#### 팀 제출 전체 삭제")
        team_all = st.text_input("팀명", key="admin_delete_team")
        if st.button("해당 팀의 모든 제출 삭제", key="btn_delete_team"):
            if not team_all.strip():
                st.warning("팀명을 입력하세요.")
            else:
                rr = requests.delete(f"{API}/admin/team/{team_all.strip()}/submissions", headers=authed_headers(), timeout=60)
                st.write(rr.status_code, rr.text)

        st.divider()

        # 섹션 3: 계정 삭제 (옵션: 같은 팀 제출도 함께 삭제)
        st.markdown("#### 계정 삭제")
        email_del = st.text_input("아이디", key="admin_delete_email")
        cascade = st.checkbox("같은 팀의 제출도 함께 삭제", value=False, key="admin_cascade")
        if st.button("계정 삭제", key="btn_delete_user"):
            if not email_del.strip():
                st.warning("아이디를 입력하세요.")
            else:
                rr = requests.delete(
                    f"{API}/admin/user/{email_del.strip()}",
                    params={"cascade_team_submissions": str(cascade).lower()},
                    headers=authed_headers(),
                    timeout=60,
                )
                st.write(rr.status_code, rr.text)


        # ==== 계정 삭제(체크박스) ====
        st.markdown("#### 계정 목록/삭제 (체크박스)")
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            user_team_filter = st.text_input("팀 필터(옵션)", key="admin_user_team_filter")
        with c2:
            user_limit = st.number_input("표시 개수", min_value=10, max_value=2000, value=20, step=10, key="admin_user_limit")
        with c3:
            if st.button("계정 목록 불러오기", key="btn_list_users"):
                try:
                    params = {"limit": int(user_limit)}
                    if user_team_filter.strip():
                        params["team"] = user_team_filter.strip()
                    r = requests.get(f"{API}/admin/users", params=params, headers=authed_headers(), timeout=30)
                    if r.status_code == 200:
                        st.session_state["_admin_users"] = pd.DataFrame(r.json())
                    else:
                        st.error(parse_err(r))
                except Exception as e:
                    st.error(str(e))

        df_users = st.session_state.get("_admin_users")
        if isinstance(df_users, pd.DataFrame) and not df_users.empty:
            work = df_users.copy()
            work["삭제"] = False
            edited = st.data_editor(
                work,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "삭제": st.column_config.CheckboxColumn("삭제", help="삭제할 계정 선택", default=False),
                    "email": st.column_config.TextColumn("아이디", disabled=True),
                    "team": st.column_config.TextColumn("팀", disabled=True),
                    "is_admin": st.column_config.CheckboxColumn("관리자", disabled=True),
                    "name": st.column_config.TextColumn("이름", disabled=True),            # ✅
                    "student_id": st.column_config.TextColumn("학번", disabled=True),      # ✅
                },
                disabled=["email","team","is_admin","name","student_id"],
            )

            colx, coly = st.columns([1,2])
            with colx:
                cascade = st.checkbox("같은 팀 제출도 삭제", value=False, key="admin_user_cascade")
            with coly:
                allow_admin_del = st.checkbox("관리자 계정도 삭제 허용", value=False, key="admin_user_allow_admin")

            if st.button("선택 계정 삭제", key="btn_delete_users"):
                target = edited.loc[edited["삭제"], ["email","is_admin","team"]]
                if target.empty:
                    st.warning("선택된 계정이 없습니다.")
                else:
                    ok, skip, fail = 0, 0, 0
                    for _, row in target.iterrows():
                        email = str(row["email"]).strip().lower()
                        is_admin_flag = bool(row["is_admin"])
                        # 관리자 보호(체크 안했으면 스킵)
                        if is_admin_flag and not allow_admin_del:
                            skip += 1
                            continue
                        try:
                            rr = requests.delete(
                                f"{API}/admin/user/{email}",
                                params={"cascade_team_submissions": str(cascade).lower()},
                                headers=authed_headers(),
                                timeout=60,
                            )
                            if rr.status_code == 200:
                                ok += 1
                            else:
                                fail += 1
                        except Exception:
                            fail += 1
                    st.success(f"삭제 완료: {ok}건, 관리자 보호로 미삭제: {skip}건, 실패: {fail}건")
                    # 목록 갱신
                    st.session_state.pop("_admin_users", None)
                    st.rerun()
        else:
            st.info("계정 목록을 불러오면 여기에서 체크박스로 삭제할 수 있어요.")
