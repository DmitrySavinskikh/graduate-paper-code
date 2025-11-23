import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure, morphology
import os

# Параметры для данных о пористости
LATENT_DIM = 128
IMG_SHAPE = (64, 64, 1)  # Увеличиваем размер для лучшего разрешения пор
BATCH_SIZE = 32
CRITIC_STEPS = 5
GP_WEIGHT = 10.0

class PorosityWGAN(keras.Model):
    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super(PorosityWGAN, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        super(PorosityWGAN, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_wass_loss_metric = keras.metrics.Mean(name="c_wass_loss")
        self.c_gp_metric = keras.metrics.Mean(name="c_gp")
        self.c_loss_metric = keras.metrics.Mean(name="c_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.c_wass_loss_metric,
            self.c_gp_metric,
            self.c_loss_metric,
            self.g_loss_metric,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Вычисление градиентного штрафа """
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # Обучение критика
        for i in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            
            with tf.GradientTape() as tape:
                fake_images = self.generator(
                    random_latent_vectors, training=True
                )
                fake_predictions = self.critic(fake_images, training=True)
                real_predictions = self.critic(real_images, training=True)
                
                c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
                c_gp = self.gradient_penalty(batch_size, real_images, fake_images)
                c_loss = c_wass_loss + c_gp * self.gp_weight

            grads = tape.gradient(c_loss, self.critic.trainable_weights)
            self.c_optimizer.apply_gradients(
                zip(grads, self.critic.trainable_weights)
            )

        # Обучение генератора
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_predictions = self.critic(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Обновление метрик
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.c_loss_metric.update_state(c_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

# Критик для пористых структур
def build_porosity_critic(img_shape):
    model = keras.Sequential([
        layers.InputLayer(input_shape=img_shape),
        
        # Первый блок
        layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        # Второй блок
        layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        # Третий блок
        layers.Conv2D(256, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        # Четвертый блок
        layers.Conv2D(512, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1)  # Линейная активация для WGAN
    ])
    return model

# Генератор для пористых структур
def build_porosity_generator(latent_dim, img_shape):
    model = keras.Sequential([
        layers.InputLayer(input_shape=(latent_dim,)),
        
        # Проекция и решейп
        layers.Dense(8 * 8 * 512),
        layers.Reshape((8, 8, 512)),
        
        # Первый блок апсемплинга
        layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Второй блок
        layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Третий блок
        layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Финальный слой
        layers.Conv2DTranspose(1, kernel_size=5, strides=1, padding="same", 
                              activation="sigmoid")  # Sigmoid для бинарных пор
    ])
    return model

# Анализ пористости
class PorosityAnalyzer:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def calculate_porosity(self, image):
        """Вычисление пористости (доли пор)"""
        binary_image = (image > self.threshold).astype(np.float32)
        porosity = np.mean(binary_image)
        return porosity
    
    def analyze_pore_structure(self, image):
        """Анализ структуры пор"""
        binary_image = (image > self.threshold).astype(np.uint8)
        
        # Анализ связанных компонентов
        labels = measure.label(binary_image, connectivity=2)
        properties = measure.regionprops(labels)
        
        pore_sizes = [prop.area for prop in properties]
        pore_count = len(pore_sizes)
        avg_pore_size = np.mean(pore_sizes) if pore_sizes else 0
        max_pore_size = np.max(pore_sizes) if pore_sizes else 0
        
        return {
            'porosity': self.calculate_porosity(image),
            'pore_count': pore_count,
            'avg_pore_size': avg_pore_size,
            'max_pore_size': max_pore_size,
            'pore_size_distribution': pore_sizes
        }

# Функции для работы с данными
def load_porosity_images(data_path, target_size=(64, 64)):
    """Загрузка и предобработка изображений пористости"""
    images = []
    
    for img_file in os.listdir(data_path):
        if img_file.endswith(('.png', '.jpg', '.jpeg', '.tiff')): # tif -> tiff
            img_path = os.path.join(data_path, img_file)
            
            # Загрузка изображения
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Изменение размера
            img = cv2.resize(img, target_size)
            
            # Нормализация
            img = img.astype(np.float32) / 255.0
            
            # Добавление размерности канала
            img = np.expand_dims(img, -1)
            images.append(img)
    
    return np.array(images)

def preprocess_porosity_data(images, binarize=False, threshold=0.5):
    """Предобработка данных пористости"""
    if binarize:
        images = (images > threshold).astype(np.float32)
    
    # Увеличение данных
    augmented_images = []
    for img in images:
        augmented_images.append(img)
        
        # Отражения
        augmented_images.append(np.fliplr(img))
        augmented_images.append(np.flipud(img))
        
        # Повороты
        for angle in [90, 180, 270]:
            rotated = np.rot90(img, k=angle//90)
            augmented_images.append(rotated)
    
    return np.array(augmented_images)

# Визуализация и анализ результатов
def analyze_and_visualize(generator, analyzer, epoch, num_samples=16):
    """Генерация и анализ образцов пористости"""
    # Генерация образцов
    test_input = tf.random.normal(shape=(num_samples, LATENT_DIM))
    generated_images = generator(test_input, training=False)
    
    # Анализ пористости
    porosity_values = []
    pore_stats = []
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = generated_images[i, :, :, 0].numpy()
        
        # Анализ
        stats = analyzer.analyze_pore_structure(img)
        porosity_values.append(stats['porosity'])
        pore_stats.append(stats)
        
        # Визуализация
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Porosity: {stats["porosity"]:.3f}\n'
                         f'Pores: {stats["pore_count"]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'porosity_generation_epoch_{epoch:04d}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Статистика пористости
    avg_porosity = np.mean(porosity_values)
    std_porosity = np.std(porosity_values)
    
    print(f"Epoch {epoch}: Average Porosity = {avg_porosity:.4f} ± {std_porosity:.4f}")
    
    return porosity_values, pore_stats

# Основная функция обучения
def train_porosity_wgan(data_path, epochs=1000):
    """Обучение WGAN для генерации пористости"""
    
    # Загрузка данных
    print("Loading porosity data...")
    images = load_porosity_images(data_path)
    print(f"Loaded {len(images)} images")
    
    # Предобработка
    processed_images = preprocess_porosity_data(images, binarize=False)
    print(f"After augmentation: {len(processed_images)} images")
    
    # Создание моделей
    critic = build_porosity_critic(IMG_SHAPE)
    generator = build_porosity_generator(LATENT_DIM, IMG_SHAPE)
    
    # Создание WGAN
    wgan = PorosityWGAN(
        critic=critic,
        generator=generator,
        latent_dim=LATENT_DIM,
        critic_steps=CRITIC_STEPS,
        gp_weight=GP_WEIGHT
    )
    
    # Компиляция
    wgan.compile(
        c_optimizer=keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9),
        g_optimizer=keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    )
    
    # Подготовка данных
    train_dataset = tf.data.Dataset.from_tensor_slices(processed_images)
    train_dataset = train_dataset.shuffle(buffer_size=2048).batch(BATCH_SIZE)
    
    # Анализатор пористости
    analyzer = PorosityAnalyzer(threshold=0.5)
    
    # Обучение
    print("Starting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        for step, real_images in enumerate(train_dataset):
            metrics = wgan.train_step(real_images)
            
            if step % 50 == 0:
                print(f"  Step {step}: {metrics}")
        
        # Анализ и визуализация каждые 50 эпох
        if (epoch + 1) % 50 == 0:
            porosity_values, pore_stats = analyze_and_visualize(
                generator, analyzer, epoch + 1
            )
            
            # Сохранение моделей
            generator.save(f'porosity_generator_epoch_{epoch+1}.h5')
            critic.save(f'porosity_critic_epoch_{epoch+1}.h5')

# Генерация новых образцов пористости
def generate_porosity_samples(generator, num_samples, target_porosity=None):
    """Генерация образцов пористости с возможным контролем пористости"""
    
    if target_porosity is not None:
        # Поиск латентных векторов, дающих нужную пористость
        samples = []
        analyzer = PorosityAnalyzer()
        
        while len(samples) < num_samples:
            noise = tf.random.normal(shape=(num_samples * 10, LATENT_DIM))
            generated = generator(noise, training=False)
            
            for img in generated:
                porosity = analyzer.calculate_porosity(img.numpy())
                if abs(porosity - target_porosity) < 0.05:  # ±5% допуск
                    samples.append(img)
                    if len(samples) >= num_samples:
                        break
        
        return tf.stack(samples[:num_samples])
    else:
        # Случайная генерация
        noise = tf.random.normal(shape=(num_samples, LATENT_DIM))
        return generator(noise, training=False)

# if __name__ == "__main__":
    # Укажите путь к вашим данным
    # data_path = "path/to/your/porosity/images"
    
    # Обучение модели
    # train_porosity_wgan(data_path, epochs=1000)
    
    # Пример генерации после обучения
    # generator = keras.models.load_model('porosity_generator_epoch_1000.h5')
    # samples = generate_porosity_samples(generator, 16, target_porosity=0.25)