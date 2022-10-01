import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
import shutil


def generate(slope, intercept):
    """
    Generates data with slope and intercept and populates dataset.csv
    """
    data_points, randomness = 200, 30
    x_data = np.linspace(0, 400, data_points)
    rand = np.random.randint(-randomness, randomness, size=data_points)
    # slope, intercept = 0.789, 60
    y_data = slope * x_data + intercept
    y_data = y_data + rand
    dataset = np.column_stack([x_data, y_data])
    df = pd.DataFrame(dataset, columns=["x", "y"])
    df.to_csv("dataset.csv")


def gif(df, x, y, m, c):
    os.makedirs("gif")
    filenames = []
    prog = 0.0
    my_bar = st.progress(prog)
    for i in range(0, len(m), 20):
        plt.scatter(df[x], df[y], c="blue")
        plt.plot(df[x], (m[i]*df[x]+c[i]))
        plt.title("Regression Plot")
        plt.xlabel("x")
        plt.ylabel("y")

        filename = f'./gif/{i}.png'
        filenames.append(filename)

        plt.savefig(filename)
        plt.close()

        my_bar.progress(prog)
        prog = prog + (1.0/len(m))*20

    with imageio.get_writer('plot.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    shutil.rmtree("gif")


def regression(df, epochs_limit, learning_rate, x, y, intercept_priority):
    m = 0
    c = 0
    m_ar = []
    c_ar = []

    losses = []
    data_len = len(df)
    loss = 100000000
    loss_dif = epochs_limit + 1
    epoch_count = 0
    log = st.empty()
    while(loss_dif > epochs_limit):
        sum_m = 0
        sum_c = 0
        dm = 0
        dc = 0
        prev_loss = loss
        for d in range(data_len):
            sum_m = sum_m + (df[x][d] * (df[y][d] - (m*df[x][d]+c)))
            sum_c = sum_c + (df[y][d] - (m*df[x][d]+c))
            loss = loss + ((df[y][d] - (m*df[x][d]+c))
                           * (df[y][d] - (m*df[x][d]+c)))
        dm = (-2/data_len)*sum_m
        dc = (-2/data_len)*sum_c * intercept_priority
        loss = loss/data_len

        m = m-learning_rate*dm
        c = c-learning_rate*dc
        losses.append(loss)

        m_ar.append(m)
        c_ar.append(c)
        loss_dif = prev_loss - loss

        log.empty()
        log.metric(label="Current Loss", value=loss, delta=-loss_dif)
        epoch_count = epoch_count+1

    st.write("Developing GIF, hold on... ")
    gif(df, x, y, m_ar, c_ar)

    return losses, m, c, epoch_count


def run(epochs_limit, intercept_priority):
    x = "x"
    y = "y"

    st.header("Dataset and Scatter Plot     ")
    df = pd.read_csv("dataset.csv", usecols=[x, y])

    col1, col2 = st.columns([1, 2])
    col1.dataframe(df)

    fig1 = plt.figure(1)
    plt.scatter(df[x], df[y])
    plt.title("Scatter Plot")
    col2.pyplot(fig1)

    st.write("***\n")

    st.header("Running Regression ")
    losses = []
    losses, m, c, epochs = regression(
        df, epochs_limit, 0.0000001, x, y, intercept_priority)
    st.write("***")

    st.header("Predictions ")
    d1, d2 = st.columns(2)
    d1.metric(label="Slope ", value=m)
    d2. metric(label="Intecept ", value=c)

    st.write("***")
    col1, col2 = st.columns(2)

    fig2 = plt.figure(2)
    plt.scatter(df[x], df[y])
    plt.plot(df[x], (m*df[x] + c))
    plt.title("Predicted Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    col1.pyplot(fig2)

    fig3 = plt.figure(3)
    plt.plot(list(range(epochs)), losses)
    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    col2.pyplot(fig3)

    st.header("Regression Line Animation")
    st.image("plot.gif")


def prep_dataset(dataset):
    df = pd.read_csv(dataset)
    x = str(st.sidebar.selectbox("Select dependent data", df.columns))
    y = str(st.sidebar.selectbox("Select independent data", df.columns))

    if st.sidebar.button("Confirm"):
        x_data = df[x]
        y_data = df[y]
        dataset = np.column_stack([x_data, y_data])
        df = pd.DataFrame(dataset, columns=["x", "y"])
        df.to_csv("dataset.csv")
        st.balloons()


def main():
    # ----------------------------- Formatting --------------------------------
    st.set_page_config(page_title="Linear Regression",
                       page_icon="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/twitter/259/mage_1f9d9.png")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; }
    footer:after {content:'Developed by Shubhayu Majumdar'; visibility: visible;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/twitter/259/mage_1f9d9.png", width=80)

    st.title("Linear Regression ")
    st.write("""
    [![Github Repo](https://img.shields.io/badge/Github%20Repo-%20-lightgrey)](https://github.com/shubhayu-64/Linear_regression_webapp)
    [![Follow](https://img.shields.io/twitter/follow/shubhayu64?style=social)](https://twitter.com/shubhayu64)
    &nbsp[![Github](https://img.shields.io/github/followers/shubhayu-64?style=social)](https://github.com/shubhayu-64)
    &nbsp[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee--yellow.svg?logo=buy-me-a-coffee&logoColor=orange&style=social)](https://www.buymeacoffee.com/shubhayu64)

    Linear regression in Python with a generated dataset.
    The dataset contains 200 data distributed evenly. Vary the slope and intercept for generated data to play around.

    This model uses __Adaptive Epochs__ over __Gradient Descent Algorithm__ based on the learning rate limit. 

    Enter your choices on the sidebar.  

    ##### PS. It may get boring at times but it works.
    ***
    """)

    if st.sidebar.selectbox("Choose type of dataset", ["Generate", "Upload"]) == "Generate":
        st.sidebar.write(
            """
            Generate a new dataset here. 
            Enter Slope and Intercept to generate random data.
            """)
        if st.sidebar.selectbox("Generate new dataset", ["Yes", "No"]) == "Yes":
            slope = float(st.sidebar.text_input("Enter slope ", 0.528))
            intercept = float(st.sidebar.text_input("Enter intercept ", 60))
            if st.sidebar.button("Generate "):
                generate(slope, intercept)
                st.balloons()
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        intercept_priority = float(st.sidebar.text_input(
            "Enter Intercept Priority parameter ", 1000))

        st.sidebar.write("***")
        epochs_limit = float(st.sidebar.text_input(
            "Enter Learning Rate Limit ", 0.001))

        if st.sidebar.button("Start Regression"):
            run(epochs_limit, intercept_priority)

    else:
        with st.sidebar.subheader('Upload your dataset'):
            dataset = st.sidebar.file_uploader(
                "Please upload a file of type: csv", type=["csv"])
        if dataset is not None:
            prep_dataset(dataset)

        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        intercept_priority = float(st.sidebar.text_input(
            "Enter Intercept Priority parameter ", 1000))

        st.sidebar.write("***")
        epochs_limit = float(st.sidebar.text_input(
            "Enter Learning Rate Limit ", 0.001))

        if st.sidebar.button("Start Regression"):
            run(epochs_limit, intercept_priority)


if __name__ == '__main__':
    main()
