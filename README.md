# Public Grievance AI - Final Version

## Main Features

### User Module
- Registration and login
- Registration confirmation email
- Submit public complaint
- Actual LSTM sentiment analysis when trained model is available
- Sentiment + confidence percentage
- Automatic complaint category
- Automatic priority
- Complaint history
- Status timeline
- In-app notifications
- Feedback after resolution

### Admin Module
- Dashboard statistics
- Registered users list
- View user and complaint history
- Edit user
- Delete user and associated complaint records
- Complaint search and filters
- AI analysis visibility
- High-priority highlighting
- Update complaint status
- Resolution / rejection remarks
- Email to citizen for every workflow status:
  Pending, Under Review, In Progress, Resolved, Rejected
- Contact Messages inbox
- Reports and charts
- CSV export

### Deep Learning
`train_lstm.py` trains an LSTM sentiment model using `dataset.csv`.
The generated files are:
- lstm_sentiment_model.keras
- tokenizer.pkl
- label_classes.json

## Python Version
Use Python 3.11.x. Recommended for this TensorFlow setup.

## First-time setup

Open CMD inside the project folder:

```cmd
py -3.11 -m pip install -r requirements.txt
```

Train the LSTM model:

```cmd
py -3.11 train_lstm.py
```

Create `.env` from `.env.example` and add Gmail SMTP/App Password settings.

Run:

```cmd
py -3.11 app.py
```

Open:

http://127.0.0.1:5000

## Default Admin
Email: admin@gmail.com
Password: admin123

Change the admin password in a real deployment.

## Email
The application sends:
1. Registration confirmation email after successful user registration.
2. Status-process email whenever an admin changes a complaint to Pending, Under Review, In Progress, Resolved or Rejected.

Complaint submission itself does not send an email, as requested.

For Gmail use a Google App Password, not the normal Gmail account password.

## Important
This is a final-year academic project. Passwords are stored simply for local demonstration. For production, use password hashing, CSRF protection, secure sessions and a production database.
