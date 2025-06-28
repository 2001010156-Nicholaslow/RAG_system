# web_ui.py
import os
import sys
import subprocess
from flask import Flask, render_template, request, redirect, url_for, Response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        permissions = request.form.getlist("permissions")
        category = request.form.get("category") or "General"

        if uploaded_file.filename == "":
            return "No file selected"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(filepath)

        return redirect(url_for("logs", filepath=filepath, category=category, permissions=",".join(permissions)))

    return render_template("index.html")


@app.route("/logs")
def logs():
    return render_template("logs.html")


@app.route("/stream")
def stream_logs():
    filepath = request.args.get("filepath")
    category = request.args.get("category")
    permissions = request.args.get("permissions").split(",")

    cmd = [
        sys.executable,
        "embedding\\embeddings.py",
        "--file", filepath,
        "--category", category,
        "--permissions", *permissions
    ]

    def generate():
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            yield f"data: {line}\n\n"
        yield "data: ✅ Done!\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
