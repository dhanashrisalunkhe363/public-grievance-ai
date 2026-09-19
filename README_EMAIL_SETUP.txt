EMAIL SETUP - PUBLIC GRIEVANCE AI

1. Install requirements:
   pip install -r requirements.txt

2. Create a file named .env inside the pgwork folder.

3. Put these lines in .env:

   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_EMAIL=yourgmail@gmail.com
   SMTP_PASSWORD=your16characterapppassword

4. Do not use quotation marks.
5. SMTP_PASSWORD must be a Gmail App Password, not your normal Gmail password.
6. Restart the project after editing .env:
   python app.py

7. Registration email is sent after a new user registers.
8. Resolution email is sent when admin changes a complaint status to Resolved.

Never share your App Password with anyone.
