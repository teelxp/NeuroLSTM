import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import psycopg2
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt


# Расширенная функция calculate_metrics
def calculate_metrics(
        true_accidents,
        predicted_accidents,
        pred_pressure,
        true_pressure,
        pred_m_out,  # <-- Предсказанный расход/масса
        m_in_test,  # <-- Входная масса (для проверки физ. баланса)
        threshold_pressure_deviation=0.1
):
    """
    Пример функции метрик:
    - Accuracy и F1 для аварий
    - Точность прогноза давления (сколько точек попало в допустимое отклонение)
    - Соответствие физическому ограничению (закон сохранения массы),
      т.е. проверяем |m_out - m_in| < threshold.
    """

    # Преобразуем в numpy
    true_acc_np = true_accidents.cpu().numpy().reshape(-1)
    pred_acc_np = predicted_accidents.cpu().numpy().reshape(-1)

    # Метрики аварий
    accuracy = accuracy_score(true_acc_np, pred_acc_np)
    f1 = f1_score(true_acc_np, pred_acc_np, zero_division=0)

    # Точность прогноза давления
    pred_pressure_np = pred_pressure.cpu().numpy().flatten()
    true_pressure_np = true_pressure.cpu().numpy().flatten()
    relative_diff = np.abs(pred_pressure_np - true_pressure_np) / (true_pressure_np + 1e-9)
    pressure_accuracy = np.mean(relative_diff < threshold_pressure_deviation) * 100.0

    # Соответствие физическому ограничению (закон сохранения массы)
    pred_m_out_np = pred_m_out.cpu().numpy().flatten()
    m_in_test_np = m_in_test.cpu().numpy().flatten()
    mass_diff = np.abs(pred_m_out_np - m_in_test_np)
    physics_consistency = np.mean(mass_diff < threshold_pressure_deviation) * 100.0

    return accuracy, f1, pressure_accuracy, physics_consistency


def load_data_from_postgres():
    """
    Загружает данные из PostgreSQL базы данных.
    """
    try:
        connection = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "training_data"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "root"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
        )
        print("Успешно подключились к базе данных.")
    except Exception as e:
        print(f"Ошибка подключения к базе данных: {e}")
        return None

    query = """
    SELECT 
        "timestamp" AS "Date and time",
        pressure,
        temperature,
        flow_rate,
        external_temp,
        humidity,
        mass_in,
        target_pressure
    FROM training_data;
    """

    try:
        df = pd.read_sql(query, connection)
        print(f"Успешно загружено {len(df)} записей из базы данных.")
    except Exception as e:
        print(f"Ошибка выполнения SQL-запроса: {e}")
        df = None
    finally:
        connection.close()

    return df


# Проверка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство для обучения: {device}")


# Адаптивная LSTM-ячейка
class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, additional_dim):
        super(AdaptiveLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.2)

        # Гейт забывания
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_f = nn.Linear(additional_dim, hidden_dim, bias=False)
        self.M_f = nn.Linear(1, hidden_dim, bias=False)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))

        # Гейт ввода
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_i = nn.Linear(additional_dim, hidden_dim, bias=False)
        self.M_i = nn.Linear(1, hidden_dim, bias=False)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))

        # Обновление состояния
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_c = nn.Linear(additional_dim, hidden_dim, bias=False)
        self.M_c = nn.Linear(1, hidden_dim, bias=False)
        self.b_c = nn.Parameter(torch.zeros(hidden_dim))

        # Выходной гейт
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_o = nn.Linear(additional_dim, hidden_dim, bias=False)
        self.M_o = nn.Linear(1, hidden_dim, bias=False)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))

        # Layer Normalization
        self.layer_norm_f = nn.LayerNorm(hidden_dim)
        self.layer_norm_i = nn.LayerNorm(hidden_dim)
        self.layer_norm_c = nn.LayerNorm(hidden_dim)
        self.layer_norm_o = nn.LayerNorm(hidden_dim)

    def forward(self, x_t, h_prev, c_prev, z_t, mass_discrepancy):
        f_t = torch.sigmoid(
            self.layer_norm_f(
                self.W_f(x_t) + self.U_f(h_prev) + self.V_f(z_t) + self.M_f(mass_discrepancy) + self.b_f
            )
        )
        i_t = torch.sigmoid(
            self.layer_norm_i(
                self.W_i(x_t) + self.U_i(h_prev) + self.V_i(z_t) + self.M_i(mass_discrepancy) + self.b_i
            )
        )
        c_hat_t = torch.tanh(
            self.layer_norm_c(
                self.W_c(x_t) + self.U_c(h_prev) + self.V_c(z_t) + self.M_c(mass_discrepancy) + self.b_c
            )
        )
        c_t = f_t * c_prev + i_t * c_hat_t
        o_t = torch.sigmoid(
            self.layer_norm_o(
                self.W_o(x_t) + self.U_o(h_prev) + self.V_o(z_t) + self.M_o(mass_discrepancy) + self.b_o
            )
        )
        h_t = self.dropout(o_t * torch.tanh(c_t))
        return h_t, c_t


# Адаптивный LSTM с тремя выходами
class AdaptiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, additional_dim, num_layers, group_dim):
        """
        Четырёхмерные входные данные:
          - (batch_size, seq_len, G, F)
            G: число групп признаков
            F: число признаков в каждой группе (после выравнивания)

        Три выхода:
          pred_pressure (канал 0),
          pred_m_out (канал 1),
          accident_logit (канал 2)
        """
        super(AdaptiveLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.group_dim = group_dim  # Новое измерение для групп признаков

        # Фактическое количество входных признаков
        total_input_dim = input_dim * group_dim

        self.lstm_cells = nn.ModuleList(
            [AdaptiveLSTMCell(total_input_dim, hidden_dim, additional_dim)]
        )
        for _ in range(1, num_layers):
            self.lstm_cells.append(AdaptiveLSTMCell(hidden_dim, hidden_dim, additional_dim))

        # Один Linear на 3 выхода
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x, z, m_in, h0=None, c0=None):
        """
        x: (batch_size, seq_len, G, F)
        z: (batch_size, seq_len, additional_dim)
        m_in: (batch_size, seq_len, 1)
        """
        batch_size, seq_len, G, F = x.size()

        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        h = [h0[layer] for layer in range(self.num_layers)]
        c = [c0[layer] for layer in range(self.num_layers)]

        outputs = []
        # Изначально полагаем, что m_out(0) = m_in(0)
        prev_m_out = m_in[:, 0, :].clone()  # (batch_size, 1)

        for t in range(seq_len):
            # Вытаскиваем x_t: (batch_size, G, F)
            x_t = x[:, t, :, :].reshape(batch_size, -1)  # -> (batch_size, G*F)

            # Доп. признаки z_t: (batch_size, additional_dim)
            z_t = z[:, t, :]

            # Масса на входе в текущий момент
            m_in_t = m_in[:, t, :]  # (batch_size, 1)

            mass_discrepancy = m_in_t - prev_m_out  # (batch_size, 1)

            # Прогон через каскад LSTM-ячеек
            hidden_input = x_t
            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.lstm_cells[layer_idx](
                    hidden_input, h[layer_idx], c[layer_idx], z_t, mass_discrepancy
                )
                hidden_input = h[layer_idx]

            out_t = self.fc(hidden_input)  # (batch_size, 3)
            outputs.append(out_t.unsqueeze(1))  # -> (batch_size, 1, 3)

            # Второй канал (index=1) считаем pred_m_out
            prev_m_out = out_t[:, 1:2].clone()  # (batch_size, 1)

        # Соединяем выходы по оси времени
        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, 3)
        return outputs, (h, c)


# Загружаем данные
df = load_data_from_postgres()
if df is None or df.empty:
    raise ValueError("Данные не были загружены из базы данных.")

# Масштабирование
scaler = MinMaxScaler()
df[["pressure", "temperature", "flow_rate",
    "external_temp", "humidity", "target_pressure", "mass_in"]] = \
    scaler.fit_transform(df[["pressure", "temperature", "flow_rate",
                             "external_temp", "humidity", "target_pressure", "mass_in"]])

# Параметры и подготовка данных
seq_len = 30
input_dim = 3  # pressure, temperature, flow_rate (каждая группа будет доведена до 3 признаков)
additional_dim = 3  # после padding у нас у группы2 тоже 3 и мы считаем external_temp + humidity + padding
group_dim = 3  # Всего 3 группы: group_1, group_2, group_3

total_samples = len(df) // seq_len
df = df.iloc[:total_samples * seq_len]

# Группа 1: (M, T, 3)
group_1 = df[["pressure", "temperature", "flow_rate"]].values.reshape(-1, seq_len, 3)

# Группа 2: (M, T, 2) -> pad до (M, T, 3)
group_2 = df[["external_temp", "humidity"]].values.reshape(-1, seq_len, 2)
group_2_padded = np.pad(group_2, ((0, 0), (0, 0), (0, 1)), mode='constant')  # -> (M, T, 3)

# Группа 3: (M, T, 1) -> pad до (M, T, 3)
group_3 = df[["mass_in"]].values.reshape(-1, seq_len, 1)
group_3_padded = np.pad(group_3, ((0, 0), (0, 0), (0, 2)), mode='constant')  # -> (M, T, 3)

# Объединяем группы (M, T, G, F=3) - четырехмерный тензор
x_data = np.stack((group_1, group_2_padded, group_3_padded), axis=2)

# Целевой признак (давление)
targets_pressure = df[["target_pressure"]].values.reshape(-1, seq_len, 1)

# Бинарная метка аварий (если pressure>0.5)
accident_labels = (df["pressure"] > 0.5).astype(int).values.reshape(-1, seq_len, 1)

# Разделение Train/Test
x_train, x_test, y_train, y_test, acc_train, acc_test = train_test_split(
    x_data,
    targets_pressure,
    accident_labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Масса на входе (3-я группа, 1-й признак после pad)
# group_3_padded -> индекс = 2, признак = [0]
m_in_train = x_train[:, :, 2, 0:1]  # (M_train, T, 1)
m_in_test = x_test[:, :, 2, 0:1]  # (M_test,  T, 1)

# Доп. признаки z для модели — возьмём группу 2 (у нас 3 признака после паддинга)
z_train = x_train[:, :, 1, :]  # (M_train, T, 3)
z_test = x_test[:, :, 1, :]  # (M_test,  T, 3)

# Преобразуем всё в тензоры (именно 4D для x!)
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)  # (M_train, T, G, F)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)  # (M_test,  T, G, F)

z_train_tensor = torch.tensor(z_train, dtype=torch.float32).to(device)  # (M_train, T, 3)
z_test_tensor = torch.tensor(z_test, dtype=torch.float32).to(device)  # (M_test,  T, 3)

m_in_train_tensor = torch.tensor(m_in_train, dtype=torch.float32).to(device)
m_in_test_tensor = torch.tensor(m_in_test, dtype=torch.float32).to(device)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

acc_train_tensor = torch.tensor(acc_train, dtype=torch.float32).to(device)
acc_test_tensor = torch.tensor(acc_test, dtype=torch.float32).to(device)

# Создаём TensorDataset
train_dataset = TensorDataset(x_train_tensor, z_train_tensor, m_in_train_tensor, y_train_tensor, acc_train_tensor)
test_dataset = TensorDataset(x_test_tensor, z_test_tensor, m_in_test_tensor, y_test_tensor, acc_test_tensor)

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Создаём модель (4D вход: (batch, seq_len, G, F))
model = AdaptiveLSTM(
    input_dim=3,  # внутри каждой группы после паддинга 3 признака
    hidden_dim=100,
    additional_dim=3,  # у нас z_train -> shape (batch, seq_len, 3)
    num_layers=3,
    group_dim=3  # всего 3 группы
).to(device)

# Функции потерь
criterion_reg = nn.L1Loss()
criterion_cls = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter('runs/adaptive_lstm_experiment_1')

num_epochs = 150
best_val_loss = np.inf

# Цикл обучения
for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for batch in train_loader:
        x_batch, z_batch, m_in_batch, y_batch, acc_batch = batch
        optimizer.zero_grad()

        # x_batch: (batch_size, seq_len, G, F) => 4D
        # z_batch: (batch_size, seq_len, 3)
        # m_in_batch: (batch_size, seq_len, 1)
        outputs, _ = model(x_batch, z_batch, m_in_batch)

        # (batch_size, seq_len, 3) на выходе
        pred_pressure = outputs[:, :, 0]
        pred_m_out = outputs[:, :, 1]
        pred_acc_logit = outputs[:, :, 2]

        y_batch = y_batch.squeeze(-1)  # (batch_size, seq_len)
        acc_batch = acc_batch.squeeze(-1)  # (batch_size, seq_len)

        loss_pressure = criterion_reg(pred_pressure, y_batch)
        loss_accident = criterion_cls(pred_acc_logit, acc_batch)
        loss_m_out = criterion_reg(pred_m_out, m_in_batch.squeeze(-1))

        total_loss = loss_pressure + loss_accident + loss_m_out
        total_loss.backward()
        optimizer.step()

        train_losses.append(total_loss.item())

    #Валидация
    model.eval()
    val_losses = []
    with torch.no_grad():
        all_preds_cls = []
        all_true_cls = []

        for batch in test_loader:
            x_batch, z_batch, m_in_batch, y_batch, acc_batch = batch
            val_outputs, _ = model(x_batch, z_batch, m_in_batch)

            val_pred_pressure = val_outputs[:, :, 0]
            val_pred_m_out = val_outputs[:, :, 1]
            val_acc_logit = val_outputs[:, :, 2]

            y_batch = y_batch.squeeze(-1)
            acc_batch = acc_batch.squeeze(-1)

            val_loss_pressure = criterion_reg(val_pred_pressure, y_batch)
            val_loss_accident = criterion_cls(val_acc_logit, acc_batch)
            val_loss = val_loss_pressure + val_loss_accident

            val_losses.append(val_loss.item())

            # Классификация аварий
            val_acc_prob = torch.sigmoid(val_acc_logit)
            val_acc_bin = (val_acc_prob > 0.5).float()
            all_preds_cls.append(val_acc_bin.cpu().numpy())
            all_true_cls.append(acc_batch.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        all_preds_cls = np.concatenate(all_preds_cls, axis=0).flatten()
        all_true_cls = np.concatenate(all_true_cls, axis=0).flatten()
        val_accuracy = accuracy_score(all_true_cls, all_preds_cls)
        val_precision = precision_score(all_true_cls, all_preds_cls, zero_division=0)
        val_recall = recall_score(all_true_cls, all_preds_cls, zero_division=0)

    # Сохраняем лучшую модель
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_adaptive_lstm.pth")
        print(f"** Лучшая модель сохранена на эпохе {epoch + 1} со значением val_loss={avg_val_loss:.4f} **")

    print(f"Эпоха {epoch + 1}/{num_epochs}: "
          f"Train Loss (MAE+BCE) = {avg_train_loss * 100:.4f}%, "
          f"Val Loss (MAE+BCE) = {avg_val_loss * 100:.4f}%, "
          f"Val Acc = {val_accuracy * 100:.2f}%, "
          f"Prec = {val_precision * 100:.2f}%, "
          f"Rec = {val_recall * 100:.2f}%")

    writer.add_scalar('Loss/train', avg_train_loss * 100, epoch)
    writer.add_scalar('Loss/val', avg_val_loss * 100, epoch)
    writer.add_scalar('Accuracy/classification', val_accuracy * 100, epoch)
    writer.add_scalar('Precision/classification', val_precision * 100, epoch)
    writer.add_scalar('Recall/classification', val_recall * 100, epoch)

writer.close()

##################################
# Тестирование лучшей модели
##################################
model.load_state_dict(torch.load("best_adaptive_lstm.pth", map_location=device))
model.eval()

with torch.no_grad():
    outputs, _ = model(x_test_tensor, z_test_tensor, m_in_test_tensor)
    # Три канала: 0=pressure, 1=m_out, 2=accident_logit
    pred_pressure = outputs[:, :, 0:1]
    pred_m_out = outputs[:, :, 1:2]
    pred_acc_logit = outputs[:, :, 2:]
    pred_acc_prob = torch.sigmoid(pred_acc_logit)
    predicted_accidents = (pred_acc_prob > 0.4).float()

    accuracy, f1, pressure_accuracy, physics_consistency = calculate_metrics(
        acc_test_tensor,
        predicted_accidents,
        pred_pressure,
        y_test_tensor,
        pred_m_out,
        m_in_test_tensor,
        threshold_pressure_deviation=0.1
    )

    print(f"Точность классификации аварий: {accuracy * 100:.2f}%")
    print(f"F1-мера классификации аварий: {f1:.4f}")
    print(f"Точность прогноза давления: {pressure_accuracy:.2f}%")
    print(f"Соответствие физическим ограничениям: {physics_consistency:.2f}%")
    print(f"Истинные аварии: {acc_test_tensor.sum().item()}, Предсказанные аварии: {predicted_accidents.sum().item()}")

    # Визуализация давления
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_tensor[0, :, 0].cpu().numpy(), label="Истинное давление")
    plt.plot(pred_pressure[0, :, 0].cpu().numpy(), label="Предсказанное давление")
    plt.xlabel("Время (шаги)")
    plt.ylabel("Давление (нормированное)")
    plt.legend()
    plt.title("Прогноз давления + проверка физ. ограничения")
    plt.grid(True)
    plt.show()


##########################################
# Визуализация Многомерных Данных
##########################################
def visualize_multidimensional_data(x, z, y, m_in, acc, sample_index=0):
    """
    Визуализирует многомерные данные для одного образца.
    """
    time_steps = np.arange(seq_len)

    fig, axs = plt.subplots(5, 1, figsize=(15, 25))

    # Группа 1: Входные признаки
    axs[0].plot(time_steps, x[sample_index, :, 0, 0].cpu().numpy(), label='Pressure')
    axs[0].plot(time_steps, x[sample_index, :, 0, 1].cpu().numpy(), label='Temperature')
    axs[0].plot(time_steps, x[sample_index, :, 0, 2].cpu().numpy(), label='Flow Rate')
    axs[0].set_title('Группа 1: Входные Признаки')
    axs[0].legend()
    axs[0].grid(True)

    # Группа 2: Дополнительные признаки
    axs[1].plot(time_steps, x[sample_index, :, 1, 0].cpu().numpy(), label='External Temp')
    axs[1].plot(time_steps, x[sample_index, :, 1, 1].cpu().numpy(), label='Humidity')
    axs[1].plot(time_steps, x[sample_index, :, 1, 2].cpu().numpy(), label='Padding')
    axs[1].set_title('Группа 2: Дополнительные Признаки')
    axs[1].legend()
    axs[1].grid(True)

    # Группа 3: Метрики
    axs[2].plot(time_steps, x[sample_index, :, 2, 0].cpu().numpy(), label='Mass In', color='orange')
    axs[2].plot(time_steps, x[sample_index, :, 2, 1].cpu().numpy(), label='Padding 1', color='gray')
    axs[2].plot(time_steps, x[sample_index, :, 2, 2].cpu().numpy(), label='Padding 2', color='lightgray')
    axs[2].set_title('Группа 3: Метрики')
    axs[2].legend()
    axs[2].grid(True)

    # Целевые переменные давления
    axs[3].plot(time_steps, y[sample_index, :, 0].cpu().numpy(), label='Target Pressure', color='green')
    axs[3].set_title('Целевые Переменные Давления')
    axs[3].legend()
    axs[3].grid(True)

    # Метки аварий
    axs[4].plot(time_steps, acc[sample_index, :, 0].cpu().numpy(), label='Accident Labels', color='red')
    axs[4].set_title('Метки Аварий')
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()


# Пример визуализации для первого образца из тестовой выборки
visualize_multidimensional_data(
    x_test_tensor,
    z_test_tensor,
    y_test_tensor,
    m_in_test_tensor,
    acc_test_tensor,
    sample_index=0
)
