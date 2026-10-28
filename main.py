from predict import main as run_predict


def main():
    """Small command-line menu for inference with the Part 2 tuned models."""
    while True:
        print("\n===== PHÂN LOẠI EMAIL SPAM =====")
        print("1. Dự đoán email mới")
        print("0. Thoát")
        choice = input("Chọn chức năng: ").strip()

        if choice == "1":
            run_predict()
        elif choice == "0":
            print("Đã thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")


if __name__ == "__main__":
    main()
