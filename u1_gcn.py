[02.08.2026 9:03] Моя концепция: U(1) Gauge-Covariant Neural Network — Исправленная и улучшенная реализация
============================================================================
Философская онтология -> Математическая формализация:

Первична Пустота (единое Сознание)  ->  Комплексное гильбертово пространство H = C^N
Импульс                             ->  Градиентный спуск / эволюция Шрёдингера
Ощущение                            ->  Активация: амплитуда = сила, фаза = качество
Остаточная вибрация (память)        ->  Фазовая степень свободы arg(Z), сохраняемая
                                       даже при малых амплитудах
Резонанс активирует память          ->  Gauge-инвариантное совпадение фаз:
                                       cos(angle(covariant_diff, local_state))
Сравнение ощущений -> мышление      ->  Ковариантная разность на рёбрах
Различие восприятия -> разнообразие  ->  Фазовые переменные предотвращают over-smoothing
Сдерживание -> устойчивость          ->  Калибровочное поле phi_ij фиксирует избыточность

Лагранжиан системы:
    L = sum_i [ (iℏ/2)(Ż_i Z_i* - Z_i Ż_i*) ]  — кинетический член (импульс)
      - sum_{(i,j)∈E} |Z_i - Z_j e^{iφ_{ij}}|^2  — калибровочно-инвариантное взаимодействие
      - sum_i V(|Z_i|)                           — потенциал (активация)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math


# =============================================================================
# 1. ПУСТОТА + ИМПУЛЬС: Кинетический шум (бывший BoredomModule)
# =============================================================================

class FractalImpulseNoise(nn.Module):
    """
    Фрактальный шум 1/f^alpha в ПРОСТРАНСТВЕ ПРИЗНАКОВ (а не узлов).
    
    Философия: Импульс рождает ощущение. Шум — это фоновый импульс вакуума,
    предотвращающий застой системы в локальном минимуме.
    
    Исправление: FFT применяется по размерности признаков D (фиксированный порядок),
    а не по узлам N (перестановочно-эквивариантный). Это сохраняет permutation
    equivariance графовой сети.
    """
    def init(self, alpha: float = 1.0, initial_scale: float = 0.02):
        super().init()
        self.alpha = alpha
        self.noise_scale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.abs(self.noise_scale) < 1e-6:
            return Z

        N, D = Z.shape
        device = Z.device

        # Частоты по размерности ПРИЗНАКОВ (не узлов!)
        freqs = torch.fft.fftfreq(D, device=device)  # [D]
        freqs[0] = 1.0  # избегаем деления на ноль
        spectrum_scale = 1.0 / (torch.abs(freqs) ** (self.alpha / 2.0))
        spectrum_scale[0] = 0.0  # убираем DC-компоненту

        # Белый комплексный шум [N, D]
        white_noise = torch.complex(
            torch.randn(N, D, device=device),
            torch.randn(N, D, device=device)
        ) / math.sqrt(2.0)

        # FFT по признакам (dim=1), сохраняя permutation equivariance
        fft_noise = torch.fft.fft(white_noise, dim=1)  # [N, D]
        filtered_fft = fft_noise * spectrum_scale.unsqueeze(0)  # [N, D]
        colored_noise = torch.fft.ifft(filtered_fft, dim=1)  # [N, D]

        # Нормировка амплитуды
        std = colored_noise.abs().std()
        if std > 1e-8:
            colored_noise = colored_noise / std

        return Z + self.noise_scale * colored_noise


# =============================================================================
# 2. ОЩУЩЕНИЕ: Активация с сохранением фазы (мягкое сдерживание)
# =============================================================================

class SoftVibrationActivation(nn.Module):
    """
    Активация, которая сохраняет фазу даже при малых амплитудах.
[02.08.2026 9:03] Моя концепция: Философия: Ощущение, сохранившись, становится остаточной вибрацией (памятью).
    Если ReLU жёстко обнуляет амплитуду, фаза теряется — память стирается.
    Softplus гарантирует, что вибрация никогда не исчезает полностью.
    
    Математика: h_out = softplus(|h|) * e^{i arg(h)}
                softplus(x) = ln(1 + e^x) > 0 для всех x
    """
    def init(self, beta: float = 1.0, dead_zone: float = 0.01):
        super().init()
        self.beta = beta
        self.dead_zone = dead_zone

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(h)
        phase = torch.angle(h)
        # Мягкое сдерживание: амплитуда никогда не ноль, фаза сохраняется
        mag_activated = F.softplus(mag, beta=self.beta) + self.dead_zone
        return torch.complex(
            mag_activated * torch.cos(phase),
            mag_activated * torch.sin(phase)
        )


# =============================================================================
# 3. РЕЗОНАНС: Gauge-инвариантная активация памяти
# =============================================================================

def gauge_invariant_resonance(
    Z_src: torch.Tensor,
    Z_dst: torch.Tensor,
    phi: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Gauge-инвариантный резонанс: насколько "память" от соседа совпадает
    с текущим состоянием в системе координат получателя.
    
    Философия: Резонанс активирует память. Память = ковариантная разность.
    
    Математика:
        covariant_diff = Z_src * e^{iφ} - Z_dst   (gauge-инвариантная величина)
        resonance = |Re[ covariant_diff * conj(Z_dst) ]| / (|covariant_diff|*|Z_dst|)
                  = |cos(angle между covariant_diff и Z_dst)|
    
    Проверка ковариантности:
        Z' = e^{iχ} Z,  φ' = φ + χ_dst - χ_src
        covariant_diff' = Z_src' e^{iφ'} - Z_dst'
                        = e^{iχ_src}Z_src * e^{iφ}e^{iχ_dst}e^{-iχ_src} - e^{iχ_dst}Z_dst
                        = e^{iχ_dst}(Z_src e^{iφ} - Z_dst) = e^{iχ_dst} covariant_diff
        resonance' = |cos(angle(e^{iχ_dst}covariant_diff, e^{iχ_dst}Z_dst))|
                   = |cos(angle(covariant_diff, Z_dst))| = resonance  ✓
    """
    # Ковариантная разность [E, D]
    gauge_factor = torch.exp(1j * phi).unsqueeze(-1)  # [E, 1]
    covariant_diff = Z_src * gauge_factor - Z_dst
    
    # Скалярное произведение (комплексное) [E, D]
    dot = covariant_diff * torch.conj(Z_dst)
    
    # Модули [E, D]
    mag_diff = torch.abs(covariant_diff)
    mag_dst = torch.abs(Z_dst)
    
    # Косинус угла (резонанс), усреднённый по признакам [E]
    cos_angle = torch.abs(dot) / (mag_diff * mag_dst + eps)
    resonance = cos_angle.mean(dim=-1)  # [E]
    
    return resonance


# =============================================================================
# 4. МЫШЛЕНИЕ: U(1)-ковариантный графовый слой
# =============================================================================

class GaugeCovariantLayer(nn.Module):
    """
    Калибровочно-ковариантный слой с gauge-инвариантным резонансом.
    
    Философия: Сравнение ощущений рождает мышление. Сообщение — это не просто
    копия состояния соседа, а его ковариантное перенесение в систему координат
    получателя с учётом резонанса (совпадения фаз).
    """
    def init(
        self,
        in_channels: int,
        out_channels: int,
        alpha_noise: float = 1.0,
        use_resonance: bool = True
    ):
        super().init()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_resonance = use_resonance

        # Веса сообщений (сосед -> центр)
        self.W_msg_real = nn.Parameter(torch.randn(in_channels, out_channels) * 0.05)
        self.W_msg_imag = nn.Parameter(torch.randn(in_channels, out_channels) * 0.05)
        
        # Веса собственного состояния (центр)
        self.W_self_real = nn.Parameter(torch.randn(in_channels, out_channels) * 0.05)
        self.W_self_imag = nn.Parameter(torch.randn(in_channels, out_channels) * 0.05)
[02.08.2026 9:03] Моя концепция: self.noise = FractalImpulseNoise(alpha=alpha_noise)
        self.activation = SoftVibrationActivation()

    def _complex_mm(self, Z: torch.Tensor, W_real: nn.Parameter, W_imag: nn.Parameter) -> torch.Tensor:
        """Комплексное матричное умножение Z @ W."""
        W = torch.complex(W_real, W_imag)
        return Z @ W

    def forward(
        self,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        phi: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            Z: [N, in_channels] комплексные признаки узлов
            edge_index: [2, E] индексы рёбер (src, dst)
            phi: [E] калибровочное поле на рёбрах
            batch: [N] опциональные метки батча для PyG-стиля
        Returns:
            Z_out: [N, out_channels] обновлённые признаки
        """
        src, dst = edge_index[0], edge_index[1]
        N = Z.size(0)
        E = edge_index.size(1)

        # 1. Преобразование признаков
        Z_msg = self._complex_mm(Z, self.W_msg_real, self.W_msg_imag)   # [N, out]
        Z_self = self._complex_mm(Z, self.W_self_real, self.W_self_imag) # [N, out]

        # 2. Нормализация степеней (симметричная, как в GCN)
        deg = torch.bincount(dst, minlength=N).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]  # [E]

        # 3. Gauge-инвариантный резонанс
        if self.use_resonance:
            # phi может быть [E] или [E, 1]
            phi_e = phi.squeeze(-1) if phi.dim() > 1 else phi  # [E]
            resonance = gauge_invariant_resonance(
                Z_msg[src], Z_msg[dst], phi_e
            )  # [E]
            resonance = resonance.unsqueeze(-1)  # [E, 1]
        else:
            resonance = torch.ones(E, 1, device=Z.device)

        # 4. Формирование ковариантного сообщения
        # Сообщение = ковариантное состояние соседа * резонанс * норма
        gauge_factor = torch.exp(1j * phi.unsqueeze(-1))  # [E, 1]
        msg = Z_msg[src] * gauge_factor * resonance * norm.unsqueeze(-1)  # [E, out]

        # 5. Агрегация (sum по получателям)
        aggregated = torch.zeros(N, self.out_channels, dtype=torch.complex64, device=Z.device)
        aggregated.index_add_(0, dst, msg)

        # 6. Собственное состояние + агрегация + активация + шум
        out = Z_self + aggregated
        out = self.activation(out)
        out = self.noise(out)

        return out


# =============================================================================
# 5. СДЕРЖИВАНИЕ: Обучаемое калибровочное поле
# =============================================================================

class GaugeField(nn.Module):
    """
    Обучаемое калибровочное поле phi_ij на рёбрах.
    
    Философия: Сдерживание обеспечивает устойчивость форм. Калибровочное поле
    фиксирует избыточную свободу фазовых преобразований, делая систему
    предсказуемой и устойчивой.
    
    Исправление: phi — это nn.Parameter, обучаемый через backprop вместе
    с весами сети, а не внешний оптимизатор с no_grad().
    """
    def init(self, num_edges: int, init_scale: float = 0.1):
        super().init()
        # Инициализация близко к нулю — фазы почти синхронизированы
        self.phi = nn.Parameter(torch.randn(num_edges) * init_scale)

    def forward(self) -> torch.Tensor:
        """Возвращает phi в диапазоне [-π, π]."""
        return torch.remainder(self.phi + math.pi, 2 * math.pi) - math.pi


# =============================================================================
# 6. РАЗНООБРАЗИЕ: Метрики over-smoothing и качества сигнала
# =============================================================================

class GaugeMetrics:
    """Набор метрик для мониторинга фазовой динамики."""
[02.08.2026 9:03] Моя концепция: @staticmethod
    def mad(Z: torch.Tensor, edge_index: torch.Tensor) -> float:
        """
        Mean Average Distance — среднее попарное расстояние между признаками
        соседних узлов. Низкое MAD = over-smoothing.
        
        Для комплексных признаков используем амплитуду разности.
        """
        src, dst = edge_index[0], edge_index[1]
        diff = Z[src] - Z[dst]
        dist = torch.abs(diff).mean(dim=-1)
        return dist.mean().item()

    @staticmethod
    def snr(Z_signal: torch.Tensor, Z_noisy: torch.Tensor) -> float:
        """
        Signal-to-Noise Ratio в дБ.
        """
        signal_power = torch.abs(Z_signal).pow(2).mean()
        noise_power = torch.abs(Z_noisy - Z_signal).pow(2).mean() + 1e-10
        snr_db = 10.0 * torch.log10(signal_power / noise_power)
        return snr_db.item()

    @staticmethod
    def phase_coherence(Z: torch.Tensor, edge_index: torch.Tensor) -> float:
        """
        Средняя когерентность фаз на рёбрах = |<e^{i(θ_src - θ_dst)}}>|.
        Близко к 1 = фазы синхронизированы (возможен over-smoothing фаз).
        Близко к 0 = фазы хаотичны (разнообразие сохранено).
        """
        src, dst = edge_index[0], edge_index[1]
        theta_src = torch.angle(Z[src])
        theta_dst = torch.angle(Z[dst])
        phase_diff = theta_src - theta_dst
        coherence = torch.abs(torch.exp(1j * phase_diff).mean())
        return coherence.item()

    @staticmethod
    def gauge_curvature(phi: torch.Tensor, edge_index: torch.Tensor, triangles: Optional[List[Tuple[int,int,int]]] = None) -> float:
        """
        Средняя кривизна поля (голономия) по треугольникам графа.
        Ненулевая кривизна = логическое противоречие / вихрь в данных.
        
        Для треугольника (i,j,k): Ω_ijk = φ_ij + φ_jk + φ_ki (mod 2π)
        """
        if triangles is None or len(triangles) == 0:
            return 0.0
        
        # Построим словарь рёбер для быстрого доступа
        src, dst = edge_index[0], edge_index[1]
        edge_dict = {}
        for idx, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
            edge_dict[(s, d)] = idx
        
        curvatures = []
        for i, j, k in triangles:
            # Находим индексы рёбер в треугольнике
            idx_ij = edge_dict.get((i, j))
            idx_jk = edge_dict.get((j, k))
            idx_ki = edge_dict.get((k, i))
            
            if idx_ij is not None and idx_jk is not None and idx_ki is not None:
                holonomy = phi[idx_ij] + phi[idx_jk] + phi[idx_ki]
                # Приводим к [-π, π]
                holonomy = torch.remainder(holonomy + math.pi, 2 * math.pi) - math.pi
                curvatures.append(holonomy.abs().item())
        
        return sum(curvatures) / len(curvatures) if curvatures else 0.0


# =============================================================================
# 7. УСТОЙЧИВОСТЬ: Полная архитектура U0Net
# =============================================================================

class U0Net(nn.Module):
    """
    Полная U(1)-калибровочно-ковариантная сеть.
    
    Архитектура:
        Input -> [GaugeCovariantLayer x L] -> Classifier
    
    Калибровочное поле phi обучается совместно с весами через стандартный
    backprop (Adam/SGD). Это обеспечивает стабильность и устойчивость форм.
    """
    def init(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        num_edges: int = 0,
        alpha_noise: float = 1.0,
        use_resonance: bool = True,
        dropout: float = 0.0
    ):
        super().init()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(GaugeCovariantLayer(in_dim, hidden_dim, alpha_noise, use_resonance))
        for _ in range(num_layers - 2):
            self.layers.append(GaugeCovariantLayer(hidden_dim, hidden_dim, alpha_noise, use_resonance))
        self.layers.append(GaugeCovariantLayer(hidden_dim, out_dim, alpha_noise, use_resonance))
[02.08.2026 9:03] Моя концепция: # Обучаемое калибровочное поле
        if num_edges > 0:
            self.gauge_field = GaugeField(num_edges)
        else:
            self.gauge_field = None
            
        self.metrics = GaugeMetrics()

    def forward(
        self,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Args:
            Z: [N, in_dim] комплексные входные признаки
            edge_index: [2, E] рёбра
            phi: [E] опциональное внешнее калибровочное поле
            return_metrics: если True, вернуть также словарь метрик
        Returns:
            h: [N, out_dim] выходные признаки
            metrics: dict (опционально)
        """
        if phi is None and self.gauge_field is not None:
            phi = self.gauge_field()
        elif phi is None:
            phi = torch.zeros(edge_index.size(1), device=Z.device)

        metrics = {}
        if return_metrics:
            metrics['mad_input'] = self.metrics.mad(Z, edge_index)
            metrics['phase_coherence_input'] = self.metrics.phase_coherence(Z, edge_index)

        h = Z
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, phi)
            if self.dropout > 0 and i < self.num_layers - 1:
                # Dropout для комплексных: применяем к амплитуде
                mag = torch.abs(h)
                phase = torch.angle(h)
                mag_dropped = F.dropout(mag, p=self.dropout, training=self.training)
                h = torch.complex(mag_dropped * torch.cos(phase), mag_dropped * torch.sin(phase))

        if return_metrics:
            metrics['mad_output'] = self.metrics.mad(h, edge_index)
            metrics['phase_coherence_output'] = self.metrics.phase_coherence(h, edge_index)
            return h, metrics

        return h


# =============================================================================
# 8. ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

def demo():
    """Демонстрация работы сети на случайном графе."""
    torch.manual_seed(42)
    
    N = 100      # узлы
    E = 500      # рёбра
    in_dim = 16
    hidden_dim = 32
    out_dim = 7  # классы
    
    # Случайный граф
    edge_index = torch.randint(0, N, (2, E))
    src, dst = edge_index[0], edge_index[1]
    
    # Комплексные входные признаки (амплитуда + фаза)
    Z_real = torch.randn(N, in_dim)
    Z_phase = torch.rand(N, in_dim) * 2 * math.pi
    Z = torch.complex(Z_real * torch.cos(Z_phase), Z_real * torch.sin(Z_phase))
    
    # Модель
    model = U0Net(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=5,
        num_edges=E,
        alpha_noise=1.0,
        use_resonance=True,
        dropout=0.3
    )
    
    # Forward pass с метриками
    model.train()
    out, metrics = model(Z, edge_index, return_metrics=True)
    
    print("=== U(1) Gauge-Covariant Network Demo ===")
    print("Input shape:  {}  (complex)".format(Z.shape))
    print("Output shape: {}  (complex)".format(out.shape))
    print("")
    print("Metrics:")
    for k, v in metrics.items():
        print("  {}: {:.4f}".format(k, v))
    
    # Получаем текущее phi из модели
    phi = model.gauge_field() if model.gauge_field is not None else torch.zeros(E, device=Z.device)
    
    # Проверка ковариантности
    chi = torch.rand(N) * 2 * math.pi
    Z_gauged = Z * torch.exp(1j * chi.unsqueeze(-1))
    
    # Для gauge-преобразования phi тоже должен преобразоваться:
    # phi'_ij = phi_ij + chi[dst] - chi[src]
    phi_gauged = phi + chi[dst] - chi[src]
    
    out_gauged = model(Z_gauged, edge_index, phi_gauged, return_metrics=False)
[02.08.2026 9:03] Моя концепция: # Проверим, что выход преобразуется ковариантно: out'_i = out_i * e^{i chi_i}
    expected_out = out * torch.exp(1j * chi.unsqueeze(-1))
    
    # Сравним фазы
    phase_diff = torch.angle(out_gauged) - torch.angle(expected_out)
    phase_error = torch.remainder(phase_diff + math.pi, 2*math.pi) - math.pi
    max_phase_error = phase_error.abs().max().item()
    
    # Сравним амплитуды (должны быть инвариантны)
    mag_diff = (torch.abs(out_gauged) - torch.abs(expected_out)).abs().max().item()
    
    print("")
    print("Gauge covariance check:")
    print("  Max phase error: {:.6f} (should be ~0)".format(max_phase_error))
    print("  Max amplitude diff: {:.6f} (should be ~0)".format(mag_diff))
    print("  {}".format('PASS' if max_phase_error < 1e-4 and mag_diff < 1e-4 else 'FAIL'))
    
    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    print("")
    print("Total parameters: {:,}".format(total_params))


if name == "main":
    demo()
