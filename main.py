from preprocess import main as run_preprocess
from train import main as run_train
from evaluate import main as run_evaluate
from predict import main as run_predict
from eda import main as run_eda


def main():
    while True:
        print("\n===== PHÂN LOẠI EMAIL SPAM =====")
        print("1. Dự đoán email mới")
        print("2. Đánh giá và so sánh hai mô hình")
        print("3. Huấn luyện hai mô hình")
        print("4. Tiền xử lý và chia dữ liệu 80/20")
        print("5. Tạo biểu đồ và thống kê EDA")
        print("0. Thoát")

        choice = input("Chọn chức năng: ").strip()

        if choice == "1":
            run_predict()

        elif choice == "2":
            run_evaluate()

        elif choice == "3":
            run_train()

        elif choice == "4":
            run_preprocess()

        elif choice == "5":
            run_eda()

        elif choice == "0":
            print("Đã thoát chương trình.")
            break

        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")


if __name__ == "__main__":
    main()
