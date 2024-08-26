import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pycountry as pc
import math
import tkinter as tk
from tkinter import ttk
import joblib
df = pd.read_csv('./em.csv')
df.isna().sum()
df.dropna(inplace=True)
x = df.drop(['Employed','Unnamed: 0'],axis=1)
y = df['Employed']
print(x)
categorical_cols = ['Age', 'Accessibility', 'EdLevel', 'Gender', 'MentalHealth', 'MainBranch', 'Country']

vectorizer = TfidfVectorizer(stop_words='english',lowercase=True)
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
transformer = ColumnTransformer([
    ('vectorizer',vectorizer,'HaveWorkedWith'),
    ('encoder', one_hot_encoder,categorical_cols)
    ])
X = transformer.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = joblib.load('model.pkl')

#model.fit(X_train,y_train)
print("Accuracy with training data:", (model.score(X_train,y_train))*100, "%")
print("Accuracy with testing data:", (model.score(X_test,y_test))*100, "%")
user_input = {
    'Age': '<35',
    'Accessibility': 'No',
    'EdLevel': 'Undergraduate',
    'Employment': 0,
    'Gender': 'Woman',
    'MentalHealth': 'Yes',
    'MainBranch': 'Dev',
    'YearsCode': 2,
    'YearsCodePro': 1,
    'Country': 'United States of America',
    'PreviousSalary': 0,
    'HaveWorkedWith': 'Assembly;C;C#;HTML/CSS;Python;SQL',
    'ComputerSkills': 6
}
#transformer.fit(x)

#X = transformer.transform(x)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#model = RandomForestClassifier()
#model.fit(X_train, y_train)

#user_df = pd.DataFrame([user_input])


class FormData:
    def __init__(self):
        self.age = None
        self.disability = None
        self.education = None
        self.employed = None
        self.gender = None
        self.depressed = None
        self.software_dev = None
        self.years_coding = None
        self.years_professional = None
        self.country = None
        self.salary = None
        self.skills = []  

        
import math
user_input['HaveWorkedWith']=''
class FormWindow(tk.Tk):
    def __init__(self, form_data=None):
        super().__init__()
        self.title("General Info")
        self.form_data = form_data if form_data else FormData()
        self.create_widgets()
    def submit(self):
        self.form_data.age= self.age_var.get()
        age=int(self.form_data.age)
        print(age)
        if age>=35:
            user_input['Age']='>35'
        else:
            user_input['Age']='<35'

        self.form_data.disability=self.dis_var.get()
        print(self.form_data.disability)
        if self.form_data.disability:
            user_input['Accessibility']='Yes'
        else:
            user_input['Accessibility']='No'

        user_input['EdLevel']=self.ed_var.get()

        user_input['Employment']=1 if self.xp_var.get()=='Yes' else 0
        gender=self.gd_var.get()
        if gender=='Male':
            user_input['Gender']='Man'
        elif gender=='Female':
            user_input['Gender']='Woman'
        else:
            user_input['Gender']='NonBinary'

        user_input['MentalHealth']='Yes' if self.mh_var.get()=='No' else 'Yes'
        user_input['MainBranch']='Dev' if self.dev_var.get()=='Yes' else 'No'
        self.form_data.years_coding = self.years_coding_var.get()
        user_input['YearsCode']=int(self.form_data.years_coding)
        user_input['YearsCodePro']=int(math.ceil(float(self.pro_var.get())))
        user_input['Country']=self.country_var.get()
        user_input['PreviousSalary']=int(self.pre_sal_val.get())
        print(user_input)
        return
    def create_widgets(self):
        
        #age
        ttk.Label(self, text="Your age:").pack(anchor="w", padx=20, pady=5)
        self.age_var = tk.StringVar()
        ttk.Spinbox(self, from_=0, to=50, width=10, textvariable=self.age_var).pack(anchor="w", padx=20)
        ttk.Label(self, text="").pack()

        #Disability
        checkbox_frame = ttk.Frame(self)
        checkbox_frame.pack(fill="x", padx=20)
        self.dis_var = tk.BooleanVar()
        dis_checkbox = ttk.Checkbutton(checkbox_frame, text="Check if you have a disability", variable=self.dis_var)
        dis_checkbox.pack(anchor="w")
        ttk.Label(self, text="").pack()


        #EdLevel
        ttk.Label(self, text="Your education level:").pack(anchor="w", padx=20)
        self.ed_var = tk.StringVar()
        ttk.Combobox(self, values=["Undergraduate", "Master", "PhD"], textvariable=self.ed_var).pack(anchor="w", padx=20)
        ttk.Label(self, text="").pack()


        #Prev xp
        xp_label = ttk.Label(self, text="Have you been employed before (Internships or full-time)?")
        xp_label.pack(anchor="w", padx=20)
        lvls = ["Yes", "No"]
        self.xp_var = tk.StringVar()
        xp_combobox = ttk.Combobox(self, values=lvls, textvariable=self.xp_var, width=15)
        xp_combobox.pack(anchor="w", padx=20)
        ttk.Label(self, text="").pack()


        #gender
        gdr_label = ttk.Label(self, text="Your Gender:")
        gdr_label.pack(anchor="w", padx=20)
        lvls = ["Male", "Female", "Other"]
        self.gd_var=tk.StringVar()
        gdr_combobox = ttk.Combobox(self, values=lvls, textvariable=self.gd_var, width=15)
        gdr_combobox.pack(anchor="w", padx=20)
        ttk.Label(self, text="").pack()

        #mental health
        mh_label = ttk.Label(self, text="Would you say you are depressed?")
        mh_label.pack(anchor="w", padx=20)
        lvls = ["Yes", "No"]
        self.mh_var=tk.StringVar()
        mh_combobox = ttk.Combobox(self, values=lvls,textvariable=self.mh_var, width=15)
        mh_combobox.pack(anchor="w", padx=20)
        ttk.Label(self, text="").pack()


        #dev
        dv_label = ttk.Label(self, text="Are you into software development?")
        dv_label.pack(anchor="w", padx=20)
        lvls = ["Yes", "No"]
        self.dev_var=tk.StringVar()
        dv_combobox = ttk.Combobox(self, values=lvls, textvariable=self.dev_var, width=15)
        dv_combobox.pack(anchor="w", padx=20)
        ttk.Label(self, text="").pack()

        # yrs input
        ttk.Label(self, text="Number of years of coding:").pack(anchor="w", padx=20, pady=5)
        self.years_coding_var = tk.StringVar()
        ttk.Spinbox(self, from_=0, to=50, width=10, textvariable=self.years_coding_var).pack(anchor="w", padx=20)
        ttk.Label(self, text="").pack()

        #pro
        pro_label = ttk.Label(self, text="Number of years of coding in a professional environment (like a job or internship):")
        pro_label.pack(anchor="w", padx=20)
        self.pro_var=tk.StringVar()
        pro_spinbox = ttk.Spinbox(self, from_=0, to=120, textvariable=self.pro_var)
        pro_spinbox.pack(anchor="w", padx=20)
        ttk.Label(self, text="").pack()

        #Countries

        cc_label = ttk.Label(self, text="Which country are you from?")
        cc_label.pack(anchor="w", padx=20)
        countries = [country.name for country in pc.countries]
        countries[countries.index('United States')]='United States of America'
        countries[countries.index('Hong Kong')]='Hong Kong (S.A.R.)'
        countries[countries.index('United Kingdom')]='United Kingdom of Great Britain and Northern Ireland'
        self.country_var=tk.StringVar()
        cc_combobox = ttk.Combobox(self, values=countries, width=50, height=20, textvariable=self.country_var)
        cc_combobox.pack(anchor="w", padx=20)
        ttk.Label(self, text="").pack()


        #prevsal
        ps_label = ttk.Label(self, text="Previous annual salary (enter 0 if you havn't worked yet):")
        ps_label.pack(anchor="w", padx=20)
        self.pre_sal_val=tk.StringVar()
        ps_spinbox = ttk.Spinbox(self, textvariable=self.pre_sal_val)
        ps_spinbox.pack(anchor="w", padx=20)
        ttk.Label(self, text="").pack()

        ttk.Button(self, text="Next", command=self.submit_and_next).pack(pady=20)

    def submit_and_next(self):
        self.submit()
        self.destroy()
        NextWindow(tk.Tk())


class NextWindow():
    def __init__(self, form_data=None):
        super().__init__()
        self.title("Programming Languages")
        self.form_data = form_data
        self.create_widgets()
    def submit(self):
        self.destroy()
        AnotherOne()

    def create_widgets(self):

        proglang=[
                "Python", "Java", "C#", "C++", "PHP", "Swift", "TypeScript", 
                "Ruby", "C", "Kotlin", "R", "Scala", "Go", "Perl", "Rust", "Dart", 
                "MATLAB", "Lua", "Haskell", "Elixir", "Clojure", "VB.NET", "F#", "Erlang", 
                "Julia", "Ada", "Fortran", "COBOL", "Scheme", "Scratch", "Prolog", "VHDL", 
                "ABAP", "Solidity", "Bash", "Groovy", "Pascal"
        ]
        ttk.Label(self, text="Select all the programming languages that you are familiar with (intermediate/advanced). These are sorted by popularity").pack(padx=10)

        tech_vars = {}
        for tech in proglang:
            var = tk.BooleanVar()
            checkbox_frame = ttk.Frame(self)
            checkbox_frame.pack(fill="x", padx=20)
            
            dis_checkbox = ttk.Checkbutton(checkbox_frame, text=tech, variable=var)
            dis_checkbox.pack(anchor="w")
            tech_vars[tech] = var
        ttk.Button(self, text="Next", command=self.submit).pack(pady=10)

        return
selected_skills = ""
class NextWindow:
    
    def __init__(self, master):
        self.master=master
        super().__init__()
        self.skills=[
                "Python", "Java", "C#", "C++", "PHP", "Swift", "TypeScript", 
                "Ruby", "C", "Kotlin", "R", "Scala", "Go", "Perl", "Rust", "Dart", 
                "MATLAB", "Lua", "Haskell", "Elixir", "Clojure", "VB.NET", "F#", "Erlang", 
                "Julia", "Ada", "Fortran", "COBOL", "Scheme", "Scratch", "Prolog", "VHDL", 
                "ABAP", "Solidity", "Bash", "Groovy", "Pascal"
        ]
        self.master.title("Programming Languages")
        self.checkboxes = []

        self.skill_vars = {}  

        self.create_widgets()
    def submit(self):

        global selected_skills
        selected_skills = ";".join([skill for skill, var in zip(self.skills, self.checkboxes) if var.get()])
        user_input['HaveWorkedWith']=user_input['HaveWorkedWith']+selected_skills+';'
        print(user_input['HaveWorkedWith'])
        self.master.destroy()
        AnotherOne(tk.Tk())
    def create_widgets(self):
        


        ttk.Label(self.master, text="Select all the programming that you are familiar with (intermediate/advanced). These are sorted by popularity").pack(padx=10)
        for skill in self.skills:
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(self.master, text=skill, variable=var)
            checkbox.pack(anchor="w", padx=20)
            self.checkboxes.append(var)

        ttk.Button(self.master, text="Next", command=self.submit).pack(pady=10)

        return

selected_skills = ""
class AnotherOne:
    
    def __init__(self, master):
        self.master=master
        super().__init__()
        self.skills=[          
            "HTML/CSS", "JavaScript", "React.js", "Angular.js", "Vue.js", "Node.js", 
            "Express.js", "Django", "Flask", "Ruby on Rails", "ASP.NET", "Laravel", 
            "Spring Boot", "jQuery", "Bootstrap", "Sass", "LESS", "TypeScript", 
            "Webpack", "Gatsby", "Next.js", "Nuxt.js", "Redux", "GraphQL", "Apollo", 
            "Tailwind CSS", "Material-UI", "Svelte", "Backbone.js", "Ember.js", 
            "Meteor", "Symfony", "CodeIgniter", "CakePHP"
        ]
        self.master.title("Web Dev technologies")
        self.checkboxes = []

        self.skill_vars = {}  

        self.create_widgets()
    def submit(self):

        global selected_skills
        selected_skills = ";".join([skill for skill, var in zip(self.skills, self.checkboxes) if var.get()])
        user_input['HaveWorkedWith']=user_input['HaveWorkedWith']+selected_skills+';'
        print(user_input['HaveWorkedWith'])
        self.master.destroy()
        AnotherTwo(tk.Tk())
    def create_widgets(self):
        


        ttk.Label(self.master, text="Select all the web dev that you are familiar with (intermediate/advanced). These are sorted by popularity").pack(padx=10)
        for skill in self.skills:
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(self.master, text=skill, variable=var)
            checkbox.pack(anchor="w", padx=20)
            self.checkboxes.append(var)

        ttk.Button(self.master, text="Next", command=self.submit).pack(pady=10)

        return

selected_skills = ""
class AnotherTwo:
    
    def __init__(self, master):
        self.master=master
        super().__init__()
        self.skills=[
                "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "Keras", "PyTorch", 
                "Matplotlib", "Seaborn", "Statsmodels", "NLTK", "SpaCy", "XGBoost", 
                "LightGBM", "CatBoost", "Gensim", "SciPy", "Plotly", "Bokeh", 
                "Theano", "OpenCV"
            ]
        self.master.title("Data Science technologies")
        self.checkboxes = []

        self.skill_vars = {}  

        self.create_widgets()
    def submit(self):

        global selected_skills
        selected_skills = ";".join([skill for skill, var in zip(self.skills, self.checkboxes) if var.get()])
        user_input['HaveWorkedWith']=user_input['HaveWorkedWith']+selected_skills+';'
        print(user_input['HaveWorkedWith'])
        self.master.destroy()
        AnotherThree(tk.Tk())
    def create_widgets(self):
        


        ttk.Label(self.master, text="Select all the data science technologies that you are familiar with (intermediate/advanced). These are sorted by popularity").pack(padx=10)
        for skill in self.skills:
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(self.master, text=skill, variable=var)
            checkbox.pack(anchor="w", padx=20)
            self.checkboxes.append(var)

        ttk.Button(self.master, text="Next", command=self.submit).pack(pady=10)

        return
selected_skills = ""
class AnotherThree:
    
    def __init__(self, master):
        self.master=master
        super().__init__()
        self.skills=[
            "MySQL", "PostgreSQL", "SQLite", "MongoDB", "Redis", "Oracle", 
            "Microsoft SQL Server", "MariaDB", "Cassandra", "Firebase", "Elasticsearch", 
            "DynamoDB", "Neo4j", "CouchDB", "HBase", "Realm", "Couchbase", "Amazon Aurora", 
            "IBM Db2", "RethinkDB"
            ]
        self.master.title("Database technologies")
        self.checkboxes = []

        self.skill_vars = {}  

        self.create_widgets()
    def submit(self):

        global selected_skills
        selected_skills = ";".join([skill for skill, var in zip(self.skills, self.checkboxes) if var.get()])
        user_input['HaveWorkedWith']=user_input['HaveWorkedWith']+selected_skills+';'
        print(user_input['HaveWorkedWith'])
        self.master.destroy()
        AnotherFour(tk.Tk())
    def create_widgets(self):
        


        ttk.Label(self.master, text="Select all the databases that you are familiar with (intermediate/advanced). These are sorted by popularity").pack(padx=10)
        for skill in self.skills:
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(self.master, text=skill, variable=var)
            checkbox.pack(anchor="w", padx=20)
            self.checkboxes.append(var)

        ttk.Button(self.master, text="Next", command=self.submit).pack(pady=10)

        return


selected_skills = ""
class AnotherFour:
    
    def __init__(self, master):
        self.master=master
        super().__init__()
        self.skills=[
                "Docker", "Kubernetes", "Git", "GitHub", "GitLab", "AWS", "Microsoft Azure", 
                "Google Cloud Platform", "Jenkins", "Terraform", "Ansible", "Azure DevOps", 
                "Bitbucket", "Travis CI", "CircleCI", "Heroku", "DigitalOcean", "Chef", 
                "Puppet", "Vagrant", "Nagios", "Prometheus", "Grafana", "Splunk", 
                "Tableau", "Power BI", "QlikView", "Apache Spark", "Hadoop", "Kafka", 
                "Apache Flink", "Airflow", "Elastic Stack (ELK)", "Consul", "Zabbix"
            ]
        self.master.title("Other technologies")
        self.checkboxes = []

        self.skill_vars = {}  

        self.create_widgets()
    def submit(self):

        global selected_skills
        selected_skills = ";".join([skill for skill, var in zip(self.skills, self.checkboxes) if var.get()])
        user_input['HaveWorkedWith']=user_input['HaveWorkedWith']+selected_skills
        print(user_input['HaveWorkedWith'])
        self.master.destroy()
        
    def create_widgets(self):
        


        ttk.Label(self.master, text="Select all the other technologies that you are familiar with (intermediate/advanced). These are sorted by popularity").pack(padx=10)
        for skill in self.skills:
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(self.master, text=skill, variable=var)
            checkbox.pack(anchor="w", padx=20)
            self.checkboxes.append(var)

        ttk.Button(self.master, text="Next", command=self.submit).pack(pady=10)

        return
   
app = FormWindow()
app.mainloop()

user_df = pd.DataFrame([user_input])

for col in x.columns:
    if col not in user_df.columns:
        user_df[col] = np.nan


user_df = user_df[x.columns]


user_X = transformer.transform(user_df)

# Make probability prediction
user_prob = model.predict_proba(user_X)

print("---------------------------------------------------------")
print(f"Probability of being employed: {user_prob[0][1]:.4f}")
ans=user_prob[0][1]*100
from PIL import Image, ImageTk
if ans>=75:
    path="1.png"
elif ans>=50:
    path="2.png"
elif ans>=25:
    path="3.png"
else:
    path="4.png"
def display_image_and_text():
    root = tk.Tk()
    root.title("Results")

    img = Image.open(path)
    
    photo = ImageTk.PhotoImage(img)

    img_label = tk.Label(root, image=photo)
    img_label.pack()

    text_label = tk.Label(root, text=f"Your probability of getting a job right now: {ans}%")
    text_label.pack()
    text_label = tk.Label(root, text=f"Accuracy of the model: {(model.score(X_test,y_test))*100}%")
    text_label.pack()

    root.mainloop()

display_image_and_text()