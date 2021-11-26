from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
dataset = pd.read_csv('courses.csv')


def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    # Get the cosine
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat


def get_recommendation(title, cosine_sim_mat, df, num_of_rec):
    course_indices = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    idx = course_indices[title]
    scores = list(enumerate(cosine_sim_mat[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sorted_scores[1:]]
    selected_course_scores = [i[1] for i in sorted_scores[1:]]
    result = df[['course_title', 'url']].iloc[selected_course_indices]
    rec_df = pd.DataFrame(result)
    rec_df['similarity_scores'] = selected_course_scores
    return rec_df.head(num_of_rec)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    df = pd.read_csv('courses.csv')
    cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])

    search_term = str(request.form.get('age'))
    num_of_rec = int(request.form.get('sex'))

    try:
        data = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)

        data = data.rename(columns={'course_title': 'Course Name', 'url': 'Course Link', 'similarity_scores': 'Similarity Score'})

        return render_template('index.html',  tables=[data.to_html(classes='data', render_links=True, escape=False, index=False)], titles=data.columns.values)


    except:
        return render_template('index.html', error="Sorry! Could Not Find This Course in the Dataset")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
