f prediction == 0:
        st.metric("Health Confidence", f"{(1-prob)*100:.1f}%")
    else:
        st.metric("Anemia Confidence", f"{prob*100:.1f}%")