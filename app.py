from flask import Flask, render_template, request, redirect, session, flash, Response
import sqlite3
import os
import pickle
import json
import smtplib
import csv
import io
from email.message import EmailMessage
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import wraps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "grievance.db")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "public-grievance-secret-change-me")

model = None
tok = None
classes = None


def now():
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%m-%Y %H:%M")

def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    c = db()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS complaints(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            complaint TEXT,
            sentiment TEXT,
            category TEXT,
            priority TEXT,
            confidence REAL,
            status TEXT DEFAULT 'Pending',
            admin_remark TEXT DEFAULT '',
            created_at TEXT,
            updated_at TEXT,
            feedback TEXT DEFAULT ''
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS contact_messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            message TEXT,
            created_at TEXT,
            status TEXT DEFAULT 'Unread'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS complaint_status_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id INTEGER,
            status TEXT,
            remark TEXT,
            changed_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS notifications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT,
            message TEXT,
            is_read INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    # Upgrade older databases safely.
    user_cols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}
    if "created_at" not in user_cols:
        c.execute("ALTER TABLE users ADD COLUMN created_at TEXT")

    complaint_cols = {r["name"] for r in c.execute("PRAGMA table_info(complaints)")}
    for col, definition in {
        "admin_remark": "TEXT DEFAULT ''",
        "updated_at": "TEXT",
        "feedback": "TEXT DEFAULT ''"
    }.items():
        if col not in complaint_cols:
            c.execute(f"ALTER TABLE complaints ADD COLUMN {col} {definition}")

    if not c.execute("SELECT id FROM users WHERE email='admin@gmail.com'").fetchone():
        c.execute(
            "INSERT INTO users(name,email,password,role,created_at) VALUES(?,?,?,?,?)",
            ("Administrator", "admin@gmail.com", "admin123", "admin", now())
        )

    c.commit()
    c.close()


def load_model():
    global model, tok, classes
    try:
        from tensorflow.keras.models import load_model
        model_path = os.path.join(BASE_DIR, "lstm_sentiment_model.keras")
        tok_path = os.path.join(BASE_DIR, "tokenizer.pkl")
        classes_path = os.path.join(BASE_DIR, "label_classes.json")
        if all(os.path.exists(x) for x in (model_path, tok_path, classes_path)):
            model = load_model(model_path)
            with open(tok_path, "rb") as f:
                tok = pickle.load(f)
            with open(classes_path, encoding="utf-8") as f:
                classes = json.load(f)
            print("Actual LSTM model loaded.")
        else:
            print("LSTM model files not found. Run: py -3.11 train_lstm.py")
    except Exception as e:
        print("LSTM unavailable:", e)


def send_email(to_email, subject, body):
    """Send email using Gmail SMTP/App Password settings from .env."""
    host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    sender = os.getenv("SMTP_EMAIL", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip().replace(" ", "")
    try:
        port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        port = 587

    if not sender or not password or not to_email:
        print("Email not configured. Check .env.")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print("Email sending failed:", e)
        return False


def predict_sentiment(text):
    if model is not None and tok is not None and classes:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = tok.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=30, padding="post")
        probabilities = model.predict(padded, verbose=0)[0]
        index = int(probabilities.argmax())
        return classes[index], round(float(probabilities[index]) * 100, 2)

    # Safe fallback so the application still runs before the model is trained.
    text_lower = text.lower()
    negative_words = [
        "bad", "not", "no", "problem", "broken", "dirty", "shortage",
        "unsafe", "unavailable", "leak", "urgent", "emergency", "danger"
    ]
    positive_words = [
        "good", "thanks", "excellent", "solved", "clean", "available", "happy"
    ]
    negative_count = sum(word in text_lower for word in negative_words)
    positive_count = sum(word in text_lower for word in positive_words)

    if negative_count > positive_count:
        return "Negative", 75.0
    if positive_count > negative_count:
        return "Positive", 75.0
    return "Neutral", 60.0


def detect_category(text):
    categories = {
        "Water Supply": ["water", "pani", "tap", "pipeline", "leak", "drinking"],
        "Road and Transport": ["road", "pothole", "traffic", "bus", "transport"],
        "Electricity": ["electricity", "power", "light", "current", "transformer"],
        "Healthcare": ["hospital", "doctor", "medicine", "health", "ambulance"],
        "Sanitation": ["garbage", "waste", "toilet", "drain", "cleanliness"],
        "Education": ["school", "college", "teacher", "education", "student"]
    }
    lower = text.lower()
    for category, words in categories.items():
        if any(word in lower for word in words):
            return category
    return "General"


def calculate_priority(sentiment, text):
    lower = text.lower()
    high_words = [
        "urgent", "emergency", "danger", "unsafe", "critical",
        "accident", "fire", "life threatening", "five days"
    ]
    if any(word in lower for word in high_words):
        return "High"
    if sentiment == "Negative":
        return "Medium"
    return "Low"


def login_required(role=None):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if "uid" not in session:
                flash("Please login first.")
                return redirect("/login")
            if role and session.get("role") != role:
                return redirect("/")
            return function(*args, **kwargs)
        return wrapper
    return decorator


@app.context_processor
def inject_context():
    unread = 0
    unread_contacts = 0
    c = db()
    if session.get("uid"):
        unread = c.execute(
            "SELECT COUNT(*) FROM notifications WHERE user_id=? AND is_read=0",
            (session["uid"],)
        ).fetchone()[0]
    if session.get("role") == "admin":
        unread_contacts = c.execute(
            "SELECT COUNT(*) FROM contact_messages WHERE status='Unread'"
        ).fetchone()[0]
    c.close()
    return {
        "current_user": session.get("name"),
        "role": session.get("role"),
        "unread_notifications": unread,
        "unread_contacts": unread_contacts
    }


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")
        created = now()

        if not name or not email or not password:
            flash("Please fill all required fields.")
            return render_template("register.html")

        c = db()

        try:

            c.execute(
                "INSERT INTO users(name,email,password,role,created_at) VALUES(?,?,?,?,?)",
                (
                    name,
                    email,
                    password,
                    "user",
                    created
                )
            )

            c.commit()

            # Get newly registered user's ID
            user = c.execute(
                "SELECT id FROM users WHERE email=?",
                (email,)
            ).fetchone()

            # Create login session
            session["uid"] = user["id"]
            session["role"] = "user"
            session["name"] = name
            session["email"] = email

            # Registration email
            send_email(
                email,
                "Public Grievance AI - Registration Successful",
                f"""Hello {name},

Your Public Grievance AI account has been created successfully.

Registered email: {email}
Registration time: {created}

You can now submit and track public complaints through the portal.

Regards,
Public Grievance AI Team"""
            )

            flash("Registration successful.")

            # Directly open User Dashboard
            return redirect("/dashboard")

        except sqlite3.IntegrityError:

            flash("This email is already registered.")

        finally:

            c.close()

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")
        c = db()
        user = c.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (email, password)
        ).fetchone()
        c.close()

        if user:
            session.update(uid=user["id"], name=user["name"], role=user["role"])
            return redirect("/admin" if user["role"] == "admin" else "/dashboard")

        flash("Invalid email or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/dashboard")
@login_required("user")
def dashboard():
    c = db()
    rows = c.execute(
        "SELECT * FROM complaints WHERE user_id=? ORDER BY id DESC",
        (session["uid"],)
    ).fetchall()
    notes = c.execute(
        "SELECT * FROM notifications WHERE user_id=? ORDER BY id DESC LIMIT 8",
        (session["uid"],)
    ).fetchall()
    c.close()

    stats = {
        "total": len(rows),
        "active": sum(r["status"] in ("Pending", "Under Review", "In Progress") for r in rows),
        "resolved": sum(r["status"] == "Resolved" for r in rows),
        "high": sum(r["priority"] == "High" for r in rows)
    }
    return render_template("dashboard.html", rows=rows, stats=stats, notifications=notes)


@app.route("/complaint", methods=["GET", "POST"])
@login_required("user")
def complaint():
    if request.method == "POST":
        text = request.form.get("complaint", "").strip()
        if len(text) < 10:
            flash("Please provide a more detailed complaint.")
            return render_template("complaint.html")

        sentiment, confidence = predict_sentiment(text)
        category = detect_category(text)
        priority = calculate_priority(sentiment, text)
        created = now()

        c = db()
        cur = c.execute(
            """INSERT INTO complaints
            (user_id,name,complaint,sentiment,category,priority,confidence,status,created_at)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                session["uid"], session["name"], text, sentiment, category,
                priority, confidence, "Pending", created
            )
        )
        complaint_id = cur.lastrowid

        c.execute(
            """INSERT INTO complaint_status_history
            (complaint_id,status,remark,changed_at)
            VALUES(?,?,?,?)""",
            (complaint_id, "Pending", "Complaint submitted and queued for review.", created)
        )

        c.execute(
            """INSERT INTO notifications(user_id,title,message,created_at)
            VALUES(?,?,?,?)""",
            (
                session["uid"],
                "Complaint Submitted",
                f"Complaint #{complaint_id} was submitted successfully and is Pending review.",
                created
            )
        )
        c.commit()
        c.close()

        return render_template(
            "result.html",
            complaint=text,
            sentiment=sentiment,
            confidence=confidence,
            category=category,
            priority=priority,
            complaint_id=complaint_id
        )

    return render_template("complaint.html")


@app.route("/admin")
@login_required("admin")
def admin():
    c = db()
    search = request.args.get("q", "").strip()
    status_filter = request.args.get("status", "").strip()
    priority_filter = request.args.get("priority", "").strip()

    query = """
        SELECT complaints.*, users.email
        FROM complaints
        LEFT JOIN users ON complaints.user_id=users.id
        WHERE 1=1
    """
    params = []

    if search:
        query += " AND (complaints.complaint LIKE ? OR complaints.name LIKE ? OR users.email LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])
    if status_filter:
        query += " AND complaints.status=?"
        params.append(status_filter)
    if priority_filter:
        query += " AND complaints.priority=?"
        params.append(priority_filter)

    query += " ORDER BY complaints.id DESC"
    rows = c.execute(query, params).fetchall()

    users = c.execute("""
        SELECT u.*, COUNT(c.id) AS complaint_count
        FROM users u
        LEFT JOIN complaints c ON u.id=c.user_id
        WHERE u.role='user'
        GROUP BY u.id
        ORDER BY u.id DESC
    """).fetchall()

    values = {}
    count_queries = {
        "total": "SELECT COUNT(*) FROM complaints",
        "positive": "SELECT COUNT(*) FROM complaints WHERE sentiment='Positive'",
        "negative": "SELECT COUNT(*) FROM complaints WHERE sentiment='Negative'",
        "neutral": "SELECT COUNT(*) FROM complaints WHERE sentiment='Neutral'",
        "high": "SELECT COUNT(*) FROM complaints WHERE priority='High'",
        "medium": "SELECT COUNT(*) FROM complaints WHERE priority='Medium'",
        "low": "SELECT COUNT(*) FROM complaints WHERE priority='Low'",
        "resolved": "SELECT COUNT(*) FROM complaints WHERE status='Resolved'",
        "active": "SELECT COUNT(*) FROM complaints WHERE status IN ('Pending','Under Review','In Progress')"
    }
    for key, query_text in count_queries.items():
        values[key] = c.execute(query_text).fetchone()[0]

    categories = c.execute(
        "SELECT category,COUNT(*) count FROM complaints GROUP BY category ORDER BY count DESC"
    ).fetchall()
    statuses = c.execute(
        "SELECT status,COUNT(*) count FROM complaints GROUP BY status"
    ).fetchall()

    c.close()
    return render_template(
        "admin.html",
        rows=rows,
        users=users,
        categories=categories,
        statuses=statuses,
        search=search,
        status_filter=status_filter,
        priority_filter=priority_filter,
        **values
    )


@app.route("/admin/user/<int:user_id>")
@login_required("admin")
def user_details(user_id):
    c = db()
    user = c.execute(
        "SELECT * FROM users WHERE id=? AND role='user'",
        (user_id,)
    ).fetchone()
    complaints = c.execute(
        "SELECT * FROM complaints WHERE user_id=? ORDER BY id DESC",
        (user_id,)
    ).fetchall()
    c.close()

    if not user:
        flash("User not found.")
        return redirect("/admin")
    return render_template("user_details.html", user=user, rows=complaints)


@app.route("/admin/user/<int:user_id>/edit", methods=["GET", "POST"])
@login_required("admin")
def edit_user(user_id):
    c = db()
    user = c.execute(
        "SELECT * FROM users WHERE id=? AND role='user'",
        (user_id,)
    ).fetchone()

    if not user:
        c.close()
        flash("User not found.")
        return redirect("/admin")

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")

        if not name or not email:
            flash("Name and email are required.")
            c.close()
            return render_template("edit_user.html", user=user)

        try:
            if password:
                c.execute(
                    "UPDATE users SET name=?,email=?,password=? WHERE id=? AND role='user'",
                    (name, email, password, user_id)
                )
            else:
                c.execute(
                    "UPDATE users SET name=?,email=? WHERE id=? AND role='user'",
                    (name, email, user_id)
                )
            c.commit()
            flash("User details updated successfully.")
            c.close()
            return redirect("/admin")
        except sqlite3.IntegrityError:
            flash("Another account already uses this email.")
            c.close()
            return render_template("edit_user.html", user=user)

    c.close()
    return render_template("edit_user.html", user=user)


@app.route("/admin/user/<int:user_id>/delete", methods=["POST"])
@login_required("admin")
def delete_user(user_id):
    c = db()
    user = c.execute(
        "SELECT * FROM users WHERE id=? AND role='user'",
        (user_id,)
    ).fetchone()

    if not user:
        c.close()
        flash("User not found.")
        return redirect("/admin")

    complaint_ids = [
        r["id"] for r in c.execute(
            "SELECT id FROM complaints WHERE user_id=?", (user_id,)
        ).fetchall()
    ]
    for complaint_id in complaint_ids:
        c.execute(
            "DELETE FROM complaint_status_history WHERE complaint_id=?",
            (complaint_id,)
        )
    c.execute("DELETE FROM complaints WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM notifications WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=? AND role='user'", (user_id,))
    c.commit()
    c.close()

    flash(f"User '{user['name']}' and associated complaint records were deleted.")
    return redirect("/admin")


def status_email_subject(status):
    return f"Public Grievance AI - Complaint Status: {status}"


def status_email_body(user_name, complaint_id, status, remark):
    process = {
        "Pending": "Your complaint is received and is waiting for official review.",
        "Under Review": "Your complaint is currently being examined by the responsible team.",
        "In Progress": "Action is currently in progress to address your complaint.",
        "Resolved": "The complaint has been marked as resolved by the administrator.",
        "Rejected": "The complaint has been marked as rejected. Please check the reason below."
    }.get(status, "Your complaint status has been updated.")

    return f"""Hello {user_name},

Your Public Grievance AI complaint #{complaint_id} has a status update.

Current Status: {status}

What is happening:
{process}

Admin Remark / Resolution:
{remark or "No additional remark was provided."}

You can login to the portal to view the complete complaint history.

Regards,
Public Grievance AI Team"""


@app.route("/update/<int:complaint_id>", methods=["POST"])
@login_required("admin")
def update_complaint(complaint_id):
    new_status = request.form.get("status", "Pending")
    remark = request.form.get("admin_remark", "").strip()
    allowed = {"Pending", "Under Review", "In Progress", "Resolved", "Rejected"}

    if new_status not in allowed:
        flash("Invalid complaint status.")
        return redirect("/admin")

    c = db()
    old = c.execute("""
        SELECT complaints.*, users.email AS user_email, users.name AS user_name
        FROM complaints
        LEFT JOIN users ON complaints.user_id=users.id
        WHERE complaints.id=?
    """, (complaint_id,)).fetchone()

    if not old:
        c.close()
        flash("Complaint not found.")
        return redirect("/admin")

    changed = old["status"] != new_status or (old["admin_remark"] or "") != remark
    changed_at = now()

    c.execute(
        "UPDATE complaints SET status=?,admin_remark=?,updated_at=? WHERE id=?",
        (new_status, remark, changed_at, complaint_id)
    )

    if changed:
        c.execute(
            """INSERT INTO complaint_status_history
            (complaint_id,status,remark,changed_at) VALUES(?,?,?,?)""",
            (complaint_id, new_status, remark, changed_at)
        )
        c.execute(
            """INSERT INTO notifications
            (user_id,title,message,created_at) VALUES(?,?,?,?)""",
            (
                old["user_id"],
                f"Complaint #{complaint_id}: {new_status}",
                f"Your complaint is now {new_status}. {remark}".strip(),
                changed_at
            )
        )

    c.commit()
    c.close()

    # Send email for every meaningful workflow status, including unresolved stages.
    if changed and old["user_email"]:
        send_email(
            old["user_email"],
            status_email_subject(new_status),
            status_email_body(old["user_name"], complaint_id, new_status, remark)
        )

    flash("Complaint updated. User notification and status email were triggered.")
    return redirect("/admin")


@app.route("/complaint/<int:complaint_id>/history")
@login_required()
def history(complaint_id):
    c = db()
    complaint_row = c.execute(
        "SELECT * FROM complaints WHERE id=?", (complaint_id,)
    ).fetchone()

    if not complaint_row:
        c.close()
        flash("Complaint not found.")
        return redirect("/dashboard" if session.get("role") == "user" else "/admin")

    if session.get("role") == "user" and complaint_row["user_id"] != session["uid"]:
        c.close()
        return redirect("/dashboard")

    history_rows = c.execute(
        "SELECT * FROM complaint_status_history WHERE complaint_id=? ORDER BY id",
        (complaint_id,)
    ).fetchall()
    c.close()
    return render_template(
        "history.html",
        complaint=complaint_row,
        history=history_rows
    )


@app.route("/complaint/<int:complaint_id>/feedback", methods=["POST"])
@login_required("user")
def feedback(complaint_id):
    value = request.form.get("feedback", "").strip()
    c = db()
    c.execute(
        "UPDATE complaints SET feedback=? WHERE id=? AND user_id=? AND status='Resolved'",
        (value, complaint_id, session["uid"])
    )
    c.commit()
    c.close()
    flash("Thank you for your feedback.")
    return redirect("/dashboard")


@app.route("/notifications")
@login_required("user")
def notifications():
    c = db()
    rows = c.execute(
        "SELECT * FROM notifications WHERE user_id=? ORDER BY id DESC",
        (session["uid"],)
    ).fetchall()
    c.execute(
        "UPDATE notifications SET is_read=1 WHERE user_id=?",
        (session["uid"],)
    )
    c.commit()
    c.close()
    return render_template("notifications.html", notifications=rows)


@app.route("/reports")
@login_required("admin")
def reports():
    c = db()
    sentiments = c.execute(
        "SELECT sentiment,COUNT(*) count FROM complaints GROUP BY sentiment"
    ).fetchall()
    priorities = c.execute(
        "SELECT priority,COUNT(*) count FROM complaints GROUP BY priority"
    ).fetchall()
    categories = c.execute(
        "SELECT category,COUNT(*) count FROM complaints GROUP BY category ORDER BY count DESC"
    ).fetchall()
    statuses = c.execute(
        "SELECT status,COUNT(*) count FROM complaints GROUP BY status"
    ).fetchall()
    daily = c.execute("""
        SELECT substr(created_at,1,10) AS day, COUNT(*) count
        FROM complaints
        GROUP BY day
        ORDER BY id DESC
        LIMIT 14
    """).fetchall()
    c.close()
    return render_template(
        "reports.html",
        sentiments=sentiments,
        priorities=priorities,
        categories=categories,
        statuses=statuses,
        daily=list(reversed(daily))
    )


@app.route("/admin/export/csv")
@login_required("admin")
def export_csv():
    c = db()
    rows = c.execute("""
        SELECT complaints.id, complaints.name, users.email, complaints.complaint,
               complaints.sentiment, complaints.confidence, complaints.category,
               complaints.priority, complaints.status, complaints.admin_remark,
               complaints.created_at, complaints.updated_at, complaints.feedback
        FROM complaints
        LEFT JOIN users ON complaints.user_id=users.id
        ORDER BY complaints.id DESC
    """).fetchall()
    c.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Complaint ID", "Citizen Name", "Email", "Complaint", "Sentiment",
        "Confidence", "Category", "Priority", "Status", "Admin Remark",
        "Created At", "Updated At", "Feedback"
    ])
    for row in rows:
        writer.writerow(list(row))

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=grievance_report.csv"}
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/admin/contacts")
@login_required("admin")
def contacts():
    c = db()
    messages = c.execute(
        "SELECT * FROM contact_messages ORDER BY id DESC"
    ).fetchall()
    c.close()
    return render_template("contacts.html", messages=messages)


@app.route("/admin/contacts/<int:message_id>/read", methods=["POST"])
@login_required("admin")
def mark_contact_read(message_id):
    c = db()
    c.execute(
        "UPDATE contact_messages SET status='Read' WHERE id=?",
        (message_id,)
    )
    c.commit()
    c.close()
    flash("Contact message marked as read.")
    return redirect("/admin/contacts")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        c = db()
        c.execute(
            """INSERT INTO contact_messages
            (name,email,message,created_at,status) VALUES(?,?,?,?,?)""",
            (
                request.form.get("name", "").strip(),
                request.form.get("email", "").lower().strip(),
                request.form.get("message", "").strip(),
                now(),
                "Unread"
            )
        )
        c.commit()
        c.close()
        flash("Your message has been submitted successfully.")
        return redirect("/contact")
    return render_template("contact.html")


if __name__ == "__main__":
    init_db()
    load_model()
    app.run(debug=True)
