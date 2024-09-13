from flask import render_template, url_for
from flask import flash, redirect, session, request
from urllib.parse import quote as url_quote
from app import app, db
from app.file_analyze import *  
from app.models import User
from app.forms import RegistrationForm, LoginForm


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data, password=form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created!', 'success')
        return redirect(url_for('login'))   #               redirect(url_for('login'))
    return render_template('register.html', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user and user.password == form.password.data:
            flash('You have been logged in!', 'success')
            return redirect(url_for('index'))  # next_page
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)


@app.route('/index') #================    NEXT PAGE MASUK KE PAGE ANALISA    ========================  methods=['GET', 'POST']
def index():  # next_page index
    if 'username' in session:
        return render_template('login')  # NEXT PAGE MASUK KE PAGE ANALISA 
    return render_template('index.html')  # redirect(url_for('index'))
    

@app.route('/finalResult', methods=['POST']) #   methods=['POST'] 
def finalResult():
    if request.method == 'POST':
        res_airlineName = request.form['airlineName']        
        selectSentiment = request.form['selectSentiment']

        # res_airlineName = request.args.get('airlineName')
        # selectSentiment = request.args.get('selectSentiment')
        
        plots , res_sentiment, most_common_words, words, frequencies = airline_func(res_airlineName, selectSentiment)
        return render_template('finalResult.html', res_airlineName= res_airlineName, selectSentiment= selectSentiment,
                                plots=plots, res_sentiment=res_sentiment, most_common_words=most_common_words,
                                words=words, frequencies=frequencies)
        


@app.route('/rules', methods=['POST'])
def rules():
    if request.method == 'POST':
        return render_template('rules.html') # redirect(url_for('rules'))
    

@app.route('/aboutUs', methods=['POST'])
def aboutUs():
    if request.method == 'POST':
        return render_template('aboutUs.html') #redirect(url_for('aboutUs')) 


@app.route("/logout", methods=['GET', 'POST'])     # ============       BACK TO MENU       =================
def logout():
    session.pop('username', None)
    # flash('You have been logged out', 'info')
    return render_template('home.html') #redirect(url_for('home'))