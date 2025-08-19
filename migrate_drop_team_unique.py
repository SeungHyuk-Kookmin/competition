import os, sqlite3
db=os.environ.get("DATABASE_URL","sqlite:///./leaderboard.db").replace("sqlite:///","")
con=sqlite3.connect(db); cur=con.cursor()
cur.execute("PRAGMA foreign_keys=off;")
cur.execute("BEGIN;")
cur.executescript("""
CREATE TABLE user_new (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL,
    team TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX uix_user_email ON user_new(email);
INSERT INTO user_new (id,email,team,password_hash,is_admin)
  SELECT id,email,team,password_hash,is_admin FROM user;
DROP TABLE user;
ALTER TABLE user_new RENAME TO user;
""")
con.commit(); con.close()
print("OK: dropped unique(team)")
