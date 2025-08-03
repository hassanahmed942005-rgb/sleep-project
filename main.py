#import libraries
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt  # FIXED: was `import matplotlib as plt`, should be `import matplotlib.pyplot as plt`

#read data
df = pd.read_csv("sleep.csv")
print(df.head())

#cleaning
df['Sleep Disorder'] = df['Sleep Disorder'].replace(np.nan, 'Normal')

#sidebar
st.sidebar.header("Sleep dashboard")
st.sidebar.image('qq.jpg')
st.sidebar.write('The purpose of this dashboard is to show the reasons for sleep disorder')

#sidebar filters
cat_filter = st.sidebar.selectbox(
    'Filters',
    ['Gender', 'Occupation', 'BMI Category', None, 'Sleep Disorder']
)

#Kpis
a1, a2, a3, a4 = st.columns(4)
a1.metric("Avg age", round(df['Age'].mean(), 2))
a2.metric("Count of ID", round(df['Person ID'].count(), 0))
a3.metric("Max daily steps", round(df['Daily Steps'].max(), 0))
a4.metric("Avg sleep duration", round(df['Sleep Duration'].mean(), 2))

# Main scatter plot
st.subheader('Sleep quality vs stress level')
fig = px.scatter(
    data_frame=df,
    x='Stress Level',
    y='Quality of Sleep',
    color=cat_filter,
    size='Quality of Sleep'
)
st.plotly_chart(fig, use_container_width=True)

# bottom charts
c1, c2 = st.columns([4, 3])

with c1:
    st.text('Occupation vs Avg Sleep Duration (Sorted)')
    avg_sleep_by_occ = df.groupby('Occupation', as_index=False)['Sleep Duration'].mean().sort_values(by='Sleep Duration', ascending=False)
    fig1 = px.bar(avg_sleep_by_occ, x="Occupation", y="Sleep Duration")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.text("Gender vs Quality of Sleep")
    gender_sleep = df.groupby('Gender', as_index=False)['Quality of Sleep'].mean()
    fig2 = px.pie(gender_sleep, names="Gender", values="Quality of Sleep")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("aaaaaaaaa")


# Pairplot for selected numeric columns
num_cols = ["Physical Activity Level", "Stress Level", "Daily Steps", "Quality of Sleep"]
st.subheader("Pair Plot for Numeric Features")
fig_pair = sns.pairplot(df[num_cols], diag_kind="kde", corner=True)
st.pyplot(fig_pair)


selected_cols= ["Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level" ,"Heart Rate","Daily Steps"]
df_selected= df[selected_cols]


# Heatmap for correlation between selected columns
st.subheader("Correlation Heatmap")
corr_matrix = df_selected.corr()

fig_heat, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)
st.pyplot(fig_heat)
