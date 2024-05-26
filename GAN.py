import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 定义生成器网络
def build_generator(latent_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=latent_dim, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='linear')  # 输出为10个元素的向量
    ])
    return model


# 定义判别器网络
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, input_shape=(10,), activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 输出为二分类（真/假）
    ])
    return model


# 定义生成对抗网络
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([
        generator,
        discriminator
    ])
    return model


# 定义训练过程
def train_gan(generator, discriminator, gan, data, latent_dim, epochs=100, batch_size=32):
    for epoch in range(epochs):
        # 生成随机噪声作为输入
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        # 生成假样本
        fake_data = generator.predict(noise)

        # 从真实数据中随机抽取样本
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]

        # 训练判别器
        discriminator_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))

        # 计算判别器总损失
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

        # 生成新的随机噪声
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # 训练生成器
        gan_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # 输出训练进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {discriminator_loss}, Generator Loss: {gan_loss}")


# 定义数据
data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

# 定义超参数
latent_dim = 100
epochs = 100
batch_size = 32

# 构建生成器、判别器和生成对抗网络
generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# 编译生成器、判别器和生成对抗网络
generator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
gan.compile(loss='binary_crossentropy', optimizer='adam')

# 训练生成对抗网络
train_gan(generator, discriminator, gan, data, latent_dim, epochs, batch_size)

# 生成新的列表
noise = np.random.normal(0, 1, (5, latent_dim))
generated_data = generator.predict(noise)
print(generated_data)