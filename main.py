import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 设置页面
st.set_page_config(
    page_title="亚高山草地分析平台",
    page_icon="🌿",
    layout="wide"
)

def main():
    st.title("🌿 亚高山草地生产力分析平台")
    st.markdown("用于分析草地生产力与环境关系的智能平台")
    
    # 初始化数据
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # 侧边栏导航
    menu = st.sidebar.selectbox("选择功能", 
                               ["首页", "数据导入", "数据分析", "模型训练", "关于"])
    
    if menu == "首页":
        show_home()
    elif menu == "数据导入":
        data_import()
    elif menu == "数据分析":
        data_analysis()
    elif menu == "模型训练":
        model_training()
    elif menu == "关于":
        show_about()

def show_home():
    st.header("欢迎使用亚高山草地分析平台")
    st.write("""
    本平台提供以下功能：
    - 📊 数据导入和管理
    - 📈 数据分析和可视化  
    - 🤖 机器学习模型训练
    - 🔮 生产力预测分析
    """)
    
    if st.button("快速开始 - 生成示例数据"):
        df = generate_sample_data()
        st.session_state.df = df
        st.success(f"成功生成 {len(df)} 条示例数据！")
        st.dataframe(df.head())

def data_import():
    st.header("数据导入")
    
    option = st.radio("选择数据来源", ["上传CSV文件", "生成示例数据"])
    
    if option == "上传CSV文件":
        uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success("数据加载成功！")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"错误: {e}")
    
    else:
        if st.button("生成示例数据"):
            df = generate_sample_data()
            st.session_state.df = df
            st.success("示例数据生成成功！")
            st.dataframe(df.head())

def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    n_samples = 150
    
    data = {
        '样地编号': [f"SP{i:03d}" for i in range(1, n_samples + 1)],
        '海拔': np.random.normal(2800, 200, n_samples),
        '坡度': np.random.normal(15, 8, n_samples),
        '土壤pH': np.random.normal(6.5, 0.5, n_samples),
        '土壤氮含量': np.random.normal(2.5, 0.8, n_samples),
        '降水量': np.random.normal(800, 150, n_samples),
        '温度': np.random.normal(8.0, 2.0, n_samples),
        '物种数': np.random.poisson(25, n_samples),
        '生物量': np.random.normal(450, 120, n_samples)
    }
    
    return pd.DataFrame(data)

def data_analysis():
    st.header("数据分析")
    
    if st.session_state.df is None:
        st.warning("请先导入数据！")
        return
        
    df = st.session_state.df
    
    st.subheader("数据概览")
    st.dataframe(df.head())
    
    st.subheader("基本统计")
    st.write(df.describe())
    
    st.subheader("数据可视化")
    
    # 散点图
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X轴", df.select_dtypes(include=np.number).columns)
    with col2:
        y_col = st.selectbox("Y轴", df.select_dtypes(include=np.number).columns)
    
    if st.button("生成散点图"):
        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col], alpha=0.6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        st.pyplot(fig)
    
    # 相关性热力图
    if st.button("显示相关性热力图"):
        numeric_cols = df.select_dtypes(include=np.number).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

def model_training():
    st.header("模型训练")
    
    if st.session_state.df is None:
        st.warning("请先导入数据！")
        return
        
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    st.subheader("选择变量")
    target = st.selectbox("目标变量（预测什么）", numeric_cols)
    features = st.multiselect("特征变量（用什么预测）", 
                             [col for col in numeric_cols if col != target])
    
    if not features:
        st.warning("请选择至少一个特征变量！")
        return
    
    model_type = st.selectbox("选择模型", ["随机森林", "线性回归"])
    
    if st.button("开始训练"):
        with st.spinner("训练模型中..."):
            try:
                X = df[features]
                y = df[target]
                
                # 处理非数值数据
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = pd.factorize(X[col])[0]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if model_type == "随机森林":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = LinearRegression()
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.success("训练完成！")
                st.metric("均方误差", f"{mse:.2f}")
                st.metric("R² 分数", f"{r2:.2f}")
                
            except Exception as e:
                st.error(f"训练失败: {e}")

def show_about():
    st.header("关于平台")
    st.write("""
    ## 亚高山草地生态系统分析平台
    
    **功能特点：**
    - 数据导入和管理
    - 统计分析
    - 可视化展示
    - 机器学习建模
    
    **技术栈：**
    - Streamlit
    - Pandas, NumPy
    - Matplotlib, Seaborn
    - Scikit-learn
    
    **适用领域：**
    - 生态学研究
    - 环境监测
    - 生产力分析
    """)

if __name__ == "__main__":
    main()