from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
dataset = pd.read_csv('courses.csv')


def vectorize_text_to_cosine_mat(data):
    vectorizer = TfidfVectorizer(max_features=1000)
    course_vectors = vectorizer.fit_transform(data.values)
    similarity = cosine_similarity(course_vectors)
    return similarity


def get_recommendation(title, similarity, df, num_of_rec):
    title = title.lower().strip(" ")
    course_index = df[df['course_title'] == title].index[0]
    scores = list(enumerate(similarity[course_index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sorted_scores[0:]]
    selected_course_scores = [i[1] for i in sorted_scores[0:]]
    result = df[['course_title', 'url']].iloc[selected_course_indices]
    rec_df = pd.DataFrame(result)
    rec_df['similarity_scores'] = selected_course_scores
    rec_df['course_title'] = rec_df['course_title'].str.title()
    return rec_df.head(num_of_rec)


@app.route('/')
def index():
    course_title = dataset['course_title'].str.title().unique()
    return render_template('index.html', course_title=course_title)



@app.route('/result', methods=['POST'])
def result():
    df = pd.read_csv('courses.csv')
    df = df.iloc[:,1:]
    course_title = dataset['course_title'].str.title().unique()
    search_term = str(request.form.get('course'))
    num_of_rec = int(request.form.get('num'))

    try:
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['content'])

        data = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)

        data = data.rename(columns={'course_title': 'Course Name', 'url': 'Course Link', 'similarity_scores': 'Similarity Score'})

        return render_template('index.html',  tables=[data.to_html(classes='data', render_links=True, escape=False, index=False)], titles=data.columns.values, course_title=course_title)


    except:
        return render_template('index.html', error="Sorry! Could Not Find This Course in the Dataset", course_title=course_title)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
