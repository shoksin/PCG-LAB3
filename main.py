import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.ndimage import generic_filter

current_image = None
image_info_label = None
hist_canvas = None


def show_temp_message(root, message, duration=2000):
    popup = tk.Toplevel(root)
    popup.overrideredirect(True)
    popup.geometry(f"400x50+{root.winfo_x() + 250}+{root.winfo_y() + 300}")
    label = tk.Label(popup, text=message, font=("Arial", 12), bg="lightgreen", fg="black", relief="solid", bd=1)
    label.pack(expand=True, fill=tk.BOTH)
    popup.after(duration, popup.destroy)


def update_image_info(file_path=None):
    global current_image, image_info_label, hist_canvas
    if current_image:
        width, height = current_image.size
        size_text = f"Размер изображения: {width}x{height}"
        if file_path:
            file_size = os.path.getsize(file_path)
            size_text += f", Вес: {file_size / 1024:.2f} KB"
        image_info_label.config(text=size_text)
        plot_color_histogram(current_image)
    else:
        image_info_label.config(text="Нет загруженного изображения.")


def plot_color_histogram(image):
    global hist_canvas
    img_array = np.array(image)
    colors = ('r', 'g', 'b') if len(img_array.shape) == 3 else ('k',)
    plt.figure(figsize=(8, 4))
    for i, color in enumerate(colors):
        channel_data = img_array[:, :, i].ravel() if len(img_array.shape) == 3 else img_array.ravel()
        plt.hist(channel_data, bins=256, color=color, alpha=0.6, label=f"{color.upper()}-канал")
    plt.title("Гистограмма цветов")
    plt.xlabel("Яркость")
    plt.ylabel("Количество пикселей")
    plt.legend(loc="upper right")
    plt.tight_layout()
    hist_canvas.delete("all")
    canvas_image = plt_to_tk()
    hist_canvas.config(width=canvas_image.width(), height=canvas_image.height())
    hist_canvas.create_image(0, 0, anchor=tk.NW, image=canvas_image)
    hist_canvas.image = canvas_image


def plt_to_tk():
    from io import BytesIO
    from PIL import ImageTk
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return ImageTk.PhotoImage(Image.open(buffer))


def load_image():
    global current_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp")])
    if file_path:
        current_image = Image.open(file_path)
        update_image_info(file_path)
        show_temp_message(root, "Изображение успешно загружено.")
    else:
        messagebox.showwarning("Ошибка", "Не удалось загрузить изображение.")


def save_image():
    global current_image
    if current_image:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            current_image.save(file_path)
            show_temp_message(root, "Изображение успешно сохранено.")
        else:
            messagebox.showwarning("Ошибка", "Не удалось сохранить изображение.")
    else:
        messagebox.showwarning("Ошибка", "Нет изображения для сохранения.")


def apply_linear_contrast():
    global current_image
    if current_image:
        enhancer = ImageEnhance.Contrast(current_image)
        current_image = enhancer.enhance(2.0)
        show_temp_message(root, "Линейное контрастирование выполнено.")
        update_image_info()
    else:
        messagebox.showwarning("Ошибка", "Сначала загрузите изображение.")


def histogram_equalization():
    global current_image
    if current_image:
        img = np.array(current_image.convert('L'))
        img_eq = cv2.equalizeHist(img)
        current_image = Image.fromarray(img_eq)
        show_temp_message(root, "Эквализация гистограммы выполнена.")
        update_image_info()
    else:
        messagebox.showwarning("Ошибка", "Сначала загрузите изображение.")


def rgb_histogram_equalization():
    global current_image
    if current_image:
        img = np.array(current_image)
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_eq = np.zeros_like(img)
        for i in range(3):
            img_eq[:, :, i] = cv2.equalizeHist(img[:, :, i])
        current_image = Image.fromarray(img_eq)
        show_temp_message(root, "Эквализация гистограммы RGB выполнена.")
        update_image_info()
    else:
        messagebox.showwarning("Ошибка", "Сначала загрузите изображение.")


def hsv_histogram_equalization():
    global current_image
    if current_image:
        img = np.array(current_image)
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_img[:, :, 2] = cv2.equalizeHist(hsv_img[:, :, 2])
        img_eq = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        current_image = Image.fromarray(img_eq)
        show_temp_message(root, "Эквализация гистограммы HSV выполнена.")
        update_image_info()
    else:
        messagebox.showwarning("Ошибка", "Сначала загрузите изображение.")


def apply_median_filter():
    global current_image
    if current_image:
        img = np.array(current_image)
        filtered_img = cv2.medianBlur(img, ksize=3)
        current_image = Image.fromarray(filtered_img)
        show_temp_message(root, "Медианный фильтр применен.")
        update_image_info()
    else:
        messagebox.showwarning("Ошибка", "Сначала загрузите изображение.")

def apply_mode_filter():
    global current_image
    if current_image:
        img = np.array(current_image)
        
        def calculate_mode(values):
            values = values.astype(np.uint8)  # Убедимся, что тип данных правильный
            return np.bincount(values).argmax()

        if len(img.shape) == 3:  # Цветное изображение (RGB)
            filtered_img = np.zeros_like(img)
            for channel in range(img.shape[2]):
                filtered_img[:, :, channel] = generic_filter(img[:, :, channel], calculate_mode, size=3)
        else:  # Чёрно-белое изображение
            filtered_img = generic_filter(img, calculate_mode, size=3)

        current_image = Image.fromarray(filtered_img.astype(np.uint8))
        show_temp_message(root, "Фильтр с модой применен.")
        update_image_info()
    else:
        messagebox.showwarning("Ошибка", "Сначала загрузите изображение.")



def show_image():
    global current_image
    if current_image:
        current_image.show()
    else:
        messagebox.showwarning("Ошибка", "Сначала загрузите изображение.")


def main():
    global root, image_info_label, hist_canvas
    root = tk.Tk()
    root.title("Обработка изображений")
    root.geometry("800x800")
    root.resizable(False, False)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=5, padx=10)

    load_save_frame = tk.Frame(button_frame)
    load_save_frame.grid(row=0, column=0, padx=10, pady=5)
    
    tk.Button(load_save_frame, text="Загрузить изображение", command=load_image, bg="lightgreen", font=("Arial", 12, "bold")).pack(pady=3)
    tk.Button(load_save_frame, text="Сохранить изображение", command=save_image, bg="lightgreen", font=("Arial", 12, "bold")).pack(pady=3)

    contrast_eq_frame = tk.Frame(button_frame)
    contrast_eq_frame.grid(row=1, column=0, padx=10, pady=3)

    tk.Button(contrast_eq_frame, text="Линейное контрастирование", command=apply_linear_contrast, bg="lightblue", font=("Arial", 12, "bold")).pack(pady=3)
    tk.Button(contrast_eq_frame, text="Эквализация гистограммы (Ч/Б)", command=histogram_equalization, bg="lightblue", font=("Arial", 12, "bold")).pack(pady=3)
    tk.Button(contrast_eq_frame, text="Эквализация гистограммы RGB", command=rgb_histogram_equalization, bg="lightblue", font=("Arial", 12, "bold")).pack(pady=3)
    tk.Button(contrast_eq_frame, text="Эквализация гистограммы HSV", command=hsv_histogram_equalization, bg="lightblue", font=("Arial", 12, "bold")).pack(pady=3)

    filter_frame = tk.Frame(button_frame)
    filter_frame.grid(row=2, column=0, padx=10, pady=3)

    tk.Button(filter_frame, text="Медианный фильтр", command=apply_median_filter, bg="lightyellow", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
    tk.Button(filter_frame, text="Фильтр с модой", command=apply_mode_filter, bg="lightyellow", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
    
    view_frame = tk.Frame(button_frame)
    view_frame.grid(row=3, column=0, padx=10, pady=3)

    tk.Button(view_frame, text="Показать изображение", command=show_image, bg="lightcoral", font=("Arial", 12, "bold")).pack(pady=3)

    info_frame = tk.Frame(root)
    info_frame.pack(side=tk.BOTTOM, pady=10)
    image_info_label = tk.Label(info_frame, text="Нет загруженного изображения.", font=("Arial", 10))
    image_info_label.pack()

    # Канвас для гистограммы
    hist_canvas = tk.Canvas(root, width=800, height=300)
    hist_canvas.pack(side=tk.BOTTOM, pady=10)

    root.mainloop()



if __name__ == "__main__":
    main()
