# Тема: "Сегментация изображения МРТ области сердца".
## Исполнители: Ромашин, Савина, Кутырёва.
## Описание
Задача состоит в том, чтобы сегментировать сердце, исходя из изображений МРТ.\
Так как изображение является медицинским, был выбран метод сегментации с помощью нейросети.

 # План проекта (техническое задание):
 1) Поиск, структуризация и подготовка данных.
 2) Освоение OpenCV, Pytorch, архитектуры нейросети Unet.
 3) Разметка данных.
 4) Построение модели. Реализация архитектуры Unet в PyTorh.
 5) Обучение модели.
 6) Оценка качества системы искусственного интеллекта.
 7) Провести сравнительный анализ характеристик оценок.

## Обязанности:
* Савина - пункты 1,2,3;
* Ромашин - пункты 2,4,5;
* Кутырёва - пункты 2,6,7.
# Пример выполнения
## Референс
<img width="1000" alt="Строение сердца" src="https://github.com/user-attachments/assets/ffa52a28-61a9-4243-9f7e-c06270bec39b">
## Входные данные
<img width="1000" alt="Снимок МРТ" src="https://github.com/user-attachments/assets/08d03995-fa8b-43cf-b7ac-6f9c18cca8ab">
## Выходные данные
 <img width="1000" alt="Снимок МРТ" src="https://github.com/user-attachments/assets/b7c01bc4-67a7-48a2-a9ff-0e0e1e7f3c66">
Здесь необходимо сделать комментарий, что это лишь примерное классификационное разграничение. Естественно, что разметка должна проходить классификационными масками.

