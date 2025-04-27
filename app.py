import streamlit as st
from PIL import Image
from pathlib import Path
import os
import shutil
from one_dcnn import main


st.markdown("""
    <style>
        body {
            background-color: #ffffff !important; 
        }

        h1, h2, h3 {
            color: #4CAF50 !important; 
            text-align: center;
        }

        .stButton > button {
            background-color: #4CAF50 !important;
            color: white !important; 
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-size: 16px !important;
        }

        .result-box {
            background-color: #E8F5E9 !important;
            border: 1px solid #4CAF50 !important; 
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .result-text {
            font-size: 20px;
            color: #4CAF50 !important; 
            text-align: center;
        }

        .recommendation-box {
            background-color: #FFF3E0 !important;
            border: 1px solid #D7CCC8 !important;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }

        .info-box {
            background-color: #E3F2FD !important; 
            border: 1px solid #BBDEFB !important; 
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }

        .stMarkdown {
            color: #333333 !important; 
        }

        .stApp {
            background-color: #F5F5F5 !important; 
        }
    </style>
""", unsafe_allow_html=True)

def get_age_range(age_group_value):
    age_ranges = {
        1: "15–19",
        2: "20–24",
        3: "25–29",
        4: "30–34",
        5: "35–39",
        6: "40–44",
        7: "45–49",
        8: "50–54",
        9: "55–59",
        10: "60–64",
        11: "65–69",
        12: "70–74",
        13: "75–79",
        14: "80–84",
        15: "85–89",
        16: "90–94",
        17: "95+"
    }
    return age_ranges.get(age_group_value, "Неизвестная категория")

st.title("Классификация возрастных категорий на основе ФПГ")
logo = Image.open("image.png")  
st.image(logo, use_container_width=True)

st.markdown("""
    <div class="info-box">
        <h3 style="text-align: center;">Как это работает?</h3>
        <p style="text-align: justify;">
            Наше приложение анализирует сигнал фотоплетизмографии (ФПГ) и определяет возрастную группу на основе извлечённых признаков. 
            Это помогает оценить биологический возраст по сравнению с паспортным.
        </p>
    </div>
""", unsafe_allow_html=True)

passport_age = st.number_input("Введите ваш паспортный возраст:", min_value=1, max_value=120, value=30, step=1)

uploaded_files = st.file_uploader("Выберите файлы (.hea и связанные файлы)", type=["hea", "dat"], accept_multiple_files=True)

def get_recommendations(predicted_age_group, passport_age):
    age_ranges = {
        1: (15, 19),
        2: (20, 24),
        3: (25, 29),
        4: (30, 34),
        5: (35, 39),
        6: (40, 44),
        7: (45, 49),
        8: (50, 54),
        9: (55, 59),
        10: (60, 64),
        11: (65, 69),
        12: (70, 74),
        13: (75, 79),
        14: (80, 84),
        15: (85, 89),
        16: (90, 94),
        17: (95, 100)
    }
    
    if predicted_age_group not in age_ranges:
        return "Неизвестная возрастная категория."
    
    min_age, max_age = age_ranges[predicted_age_group]
    
    if min_age <= passport_age <= max_age:
        return """
            <div style="text-align: center; font-size: 16px; color: #4CAF50;">
                Ваш паспортный возраст находится в пределах предсказанного диапазона. 
                Это хороший знак! Однако регулярное медицинское обследование всегда полезно.
            </div>
        """
    elif abs(passport_age - (min_age + max_age) / 2) <= 3:
        return """
            <div style="text-align: center; font-size: 16px; color: #FF9800;">
                Ваш паспортный возраст немного отличается от предсказанного диапазона. 
                Это может быть связано с особенностями организма. Рекомендуется проконсультироваться с врачом.
            </div>
        """
    else:
        return """
            <div style="text-align: center; font-size: 16px; color: #F44336;">
                Значительная разница между предсказанным и паспортным возрастом. 
                Это повод обратиться к врачу для дополнительного обследования.
            </div>
        """

if uploaded_files:
    try:
        temp_dir = Path("temp_data")
        temp_dir.mkdir(exist_ok=True)
        
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            file_paths.append(file_path)
        
        hea_file = None
        for file_path in file_paths:
            if file_path.suffix == ".hea":
                hea_file = file_path
                break
        
        if not hea_file:
            st.error("Не найден файл .hea среди загруженных файлов.")
        else:
            prediction = main(str(hea_file))
            
            predicted_age_group = int(prediction.split(":")[1].strip())
            
            age_range = get_age_range(predicted_age_group)
            
            st.markdown(f"""
                <div class="result-box">
                    <div class="result-text">Предсказанная возрастная группа: {predicted_age_group} ({age_range})</div>
                </div>
            """, unsafe_allow_html=True)
            
            recommendations = get_recommendations(predicted_age_group, passport_age)
            st.markdown(recommendations, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Произошла ошибка при обработке файлов: {e}")
    
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

st.markdown("""
    <div class="info-box">
        <h3 style="text-align: center;">Почему важно следить за здоровьем?</h3>
        <p style="text-align: justify;">
            Биологический возраст может отличаться от паспортного. Это зависит от образа жизни, питания, стресса и других факторов. 
            Регулярные медицинские обследования помогают выявить проблемы на ранних этапах и сохранить здоровье.
        </p>
    </div>
""", unsafe_allow_html=True)
