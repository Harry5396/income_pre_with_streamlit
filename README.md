income_app_streamlit

Đây là ứng dụng dự đoán income theo các input đầu vào, được triển khai lên nền tảng streamlit

Để chạy trên máy local của bạn:
1. clone tất cả file về
2. Chạy notebook để train mô hình --> lưu mô hình lại với tên chính xác: "income_pre.pkl"
3. lưu file model income_pre.pkl về thư mục chứa file Icome_pre.py
4. Cài đặt các thư viện cần thiết
5. chạy cmd: streamlit run Income_pre.py

Kết quả sẽ được như hình sau:
![Screenshot 2023-03-01 at 18 06 36](https://user-images.githubusercontent.com/75346165/222122230-13b0889c-62d2-4803-969e-41a0f460e27f.png)



Mô hình được đào tạo bằng dữ liệu trên kaggle: 
# https://www.kaggle.com/datasets/uciml/adult-census-income
