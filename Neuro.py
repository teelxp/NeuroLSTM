import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

# Проверка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство для обучения: {device}")


# Адаптивная (модифицированная) LSTM-ячейка
class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, additional_dim):
        super(AdaptiveLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.2)  # Dropout для регуляризации

        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_f = nn.Linear(additional_dim, hidden_dim, bias=False)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))

        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_i = nn.Linear(additional_dim, hidden_dim, bias=False)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))

        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_c = nn.Linear(additional_dim, hidden_dim, bias=False)
        self.b_c = nn.Parameter(torch.zeros(hidden_dim))

        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_o = nn.Linear(additional_dim, hidden_dim, bias=False)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_t, h_prev, c_prev, z_t):
        f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev) + self.V_f(z_t) + self.b_f)
        i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev) + self.V_i(z_t) + self.b_i)
        c_hat_t = torch.tanh(self.W_c(x_t) + self.U_c(h_prev) + self.V_c(z_t) + self.b_c)
        c_t = f_t * c_prev + i_t * c_hat_t
        o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev) + self.V_o(z_t) + self.b_o)
        h_t = self.dropout(o_t * torch.tanh(c_t))  # Dropout на выходе
        return h_t, c_t


# Полный адаптивный (модифицированный) LSTM-слой
class AdaptiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, additional_dim, num_layers, output_dim=3):
        super(AdaptiveLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm_cells = nn.ModuleList(
            [AdaptiveLSTMCell(input_dim, hidden_dim, additional_dim)]
        )
        for _ in range(1, num_layers):
            self.lstm_cells.append(AdaptiveLSTMCell(hidden_dim, hidden_dim, additional_dim))

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, z, h0=None, c0=None):
        batch_size, seq_len, _ = x.size()

        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        h, c = [h0[layer] for layer in range(self.num_layers)], [c0[layer] for layer in range(self.num_layers)]
        outputs = []

        for t in range(seq_len):
            x_t, z_t = x[:, t, :], z[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](x_t, h[layer], c[layer], z_t)
                x_t = h[layer]
            outputs.append(h[-1])

        outputs = torch.stack(outputs, dim=1)
        return self.fc(outputs), (torch.stack(h), torch.stack(c))


# Модифицированная функция потерь
def modified_loss(pred_pressure, true_pressure, m_in, m_out_pred, lambda_val=1.0):
    data_loss = F.mse_loss(pred_pressure, true_pressure)
    physics_loss = F.mse_loss(m_in, m_out_pred)
    return data_loss + lambda_val * physics_loss, data_loss, physics_loss


# Нормализация данных
data_file = "data_for_lstm.csv"
df = pd.read_csv(data_file)
scaler = MinMaxScaler()
df[["pressure", "temperature", "flow_rate", "external_temp", "humidity", "target_pressure", "mass_in"]] = \
    scaler.fit_transform(
        df[["pressure", "temperature", "flow_rate", "external_temp", "humidity", "target_pressure", "mass_in"]])

# Настройка параметров
seq_len, input_dim, additional_dim, output_dim = 30, 3, 2, 3
total_samples = len(df) // seq_len
df = df.iloc[:total_samples * seq_len]

x_data = df[["pressure", "temperature", "flow_rate"]].values.reshape(-1, seq_len, input_dim)
z_data = df[["external_temp", "humidity"]].values.reshape(-1, seq_len, additional_dim)
targets_pressure = df[["target_pressure"]].values.reshape(-1, seq_len, 1)
m_in = df[["mass_in"]].values.reshape(-1, seq_len, 1)
accident_labels = (df["pressure"] > 0.5).astype(int).values.reshape(-1, seq_len, 1)

x_train, x_test, z_train, z_test, y_train, y_test, m_in_train, m_in_test, acc_train, acc_test = train_test_split(
    x_data, z_data, targets_pressure, m_in, accident_labels, test_size=0.2, random_state=42
)

to_tensor = lambda *arrays: [torch.tensor(arr, dtype=torch.float32).to(device) for arr in arrays]
x_train, x_test, z_train, z_test, y_train, y_test, m_in_train, m_in_test, acc_train, acc_test = to_tensor(
    x_train, x_test, z_train, z_test, y_train, y_test, m_in_train, m_in_test, acc_train, acc_test
)

# Аугментация аварий
accidents = x_train[acc_train.squeeze(-1) == 1]  # Примеры аварий (класс 1)
acc_z = z_train[acc_train.squeeze(-1) == 1]
acc_y = y_train[acc_train.squeeze(-1) == 1]
acc_m_in = m_in_train[acc_train.squeeze(-1) == 1]

if accidents.size(0) > 0:  # Проверяем, есть ли аварии
    noise = torch.normal(0, 0.01, size=accidents.size(), device=device)
    augmented_accidents = accidents + noise

    # Проверяем размерности и корректируем
    if augmented_accidents.size(1) != seq_len:
        # Разбиваем данные на последовательности длины seq_len
        augmented_accidents = augmented_accidents.unfold(0, seq_len, seq_len).permute(0, 2, 1)

    if acc_z.size(1) != seq_len:
        acc_z = acc_z.unfold(0, seq_len, seq_len).permute(0, 2, 1)
    if acc_y.size(1) != seq_len:
        acc_y = acc_y.unfold(0, seq_len, seq_len).permute(0, 2, 1)
    if acc_m_in.size(1) != seq_len:
        acc_m_in = acc_m_in.unfold(0, seq_len, seq_len).permute(0, 2, 1)

    # Добавляем аугментированные данные
    x_train = torch.cat([x_train, augmented_accidents], dim=0)
    z_train = torch.cat([z_train, acc_z], dim=0)
    y_train = torch.cat([y_train, acc_y], dim=0)
    m_in_train = torch.cat([m_in_train, acc_m_in], dim=0)
    acc_train = torch.cat([acc_train, torch.ones_like(augmented_accidents[:, :, 0:1])], dim=0)

# Проверка размеров данных после аугментации
print(f"x_train shape: {x_train.shape}")
print(f"z_train shape: {z_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"m_in_train shape: {m_in_train.shape}")
print(f"acc_train shape: {acc_train.shape}")


# Модель
model = AdaptiveLSTM(input_dim, 50, additional_dim, 2, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.01)
classification_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))

# Функция склонения слов
def get_plural_form(value, singular, dual, plural):
    if 11 <= value % 100 <= 19:
        return plural
    elif value % 10 == 1:
        return singular
    elif 2 <= value % 10 <= 4:
        return dual
    else:
        return plural

# Измерение времени
start_time = time.time()

# Обучение
epochs, lambda_val = 25000, 10.0
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs, _ = model(x_train, z_train)
    pred_pressure, pred_m_out, pred_accident = outputs[:, :, 0:1], outputs[:, :, 1:2], outputs[:, :, 2:]

    loss_regression, data_loss, physics_loss = modified_loss(pred_pressure, y_train, m_in_train, pred_m_out, lambda_val)
    loss_classification = classification_loss_fn(pred_accident, acc_train)
    total_loss = loss_regression + lambda_val * physics_loss + 15 * loss_classification

    total_loss.backward()
    optimizer.step()

    # Шаг изменения скорости обучения
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Эпоха {epoch + 1}, Общая ошибка: {total_loss.item():.4f}, "
            f"Ошибка данных: {data_loss.item():.4f}, Ошибка физического ограничения: {physics_loss.item():.4f}, "
            f"Ошибка классификации: {loss_classification.item():.4f}"
        )

        # Текущая скорость обучения
        current_lr = scheduler.get_last_lr()[0]
        print(f"Текущая скорость обучения: {current_lr:.6f}")

# Рассчёт времени выполнения
end_time = time.time()
training_time_seconds = end_time - start_time

training_time_hours = int(training_time_seconds // 3600)
training_time_minutes = int((training_time_seconds % 3600) // 60)
remaining_seconds = int(training_time_seconds % 60)

hours_word = get_plural_form(training_time_hours, "час", "часа", "часов")
minutes_word = get_plural_form(training_time_minutes, "минута", "минуты", "минут")
seconds_word = get_plural_form(remaining_seconds, "секунда", "секунды", "секунд")

print(
    f"Общее время обучения: {training_time_hours} {hours_word}, "
    f"{training_time_minutes} {minutes_word}, {remaining_seconds} {seconds_word}."
)

# Тестирование и прогноз аварий
model.eval()
with torch.no_grad():
    outputs, _ = model(x_test, z_test)
    pred_accident = torch.sigmoid(outputs[:, :, 2:])  # Вероятности аварий

    predicted_accidents = (pred_accident > 0.5).float()
    print(f"Истинные аварии: {acc_test.sum().item()}, Предсказанные аварии: {predicted_accidents.sum().item()}")

    # Сообщение о предсказанных авариях
    if predicted_accidents.sum().item() > 0:
        print("Модель предсказывает, что возможны аварии.")
    else:
        print("Модель не предсказывает аварий.")

    # Визуализация
    plt.plot(y_test[0, :, 0].cpu().numpy(), label="Истинное давление")
    plt.plot(outputs[0, :, 0].cpu().numpy(), label="Предсказанное давление")
    plt.legend()
    plt.title("Прогноз давления на тестовых данных")
    plt.show()
