from flask import Flask,render_template,request,redirect
from speakerDiarization import main

app=Flask(__name__)



@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
            
        if file:
            main(file)

    return render_template('index.html')


app.run(debug=True)