//! Módulo de análise de séries temporais com regressão linear
//! Implementação pura sem dependências externas

use std::fmt;
use std::error::Error;

/// Estrutura para armazenar os resultados da regressão linear
#[derive(Debug, Clone, PartialEq)]
pub struct LinearRegressionResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub mse: f64,
    pub predictions: Vec<f64>,
}

/// Estrutura para representar erros na análise de séries temporais
#[derive(Debug, Clone)]
pub struct TimeSeriesError {
    message: String,
}

impl TimeSeriesError {
    pub fn new(msg: &str) -> Self {
        TimeSeriesError {
            message: msg.to_string(),
        }
    }
}

impl fmt::Display for TimeSeriesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TimeSeriesError: {}", self.message)
    }
}

impl Error for TimeSeriesError {}

/// Realiza regressão linear em uma série temporal
pub fn linear_regression(data: &[f64]) -> Result<LinearRegressionResult, TimeSeriesError> {
    if data.len() < 2 {
        return Err(TimeSeriesError::new("Dados insuficientes para regressão linear"));
    }

    let n = data.len() as f64;
    let x: Vec<f64> = (0..data.len()).map(|x| x as f64).collect();
    
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = data.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..data.len() {
        numerator += (x[i] - x_mean) * (data[i] - y_mean);
        denominator += (x[i] - x_mean).powi(2);
    }
    
    let slope = if denominator.abs() < f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    };
    
    let intercept = y_mean - slope * x_mean;
    
    let predictions: Vec<f64> = x.iter().map(|&xi| intercept + slope * xi).collect();
    let mse = calculate_mse(data, &predictions);
    let r_squared = calculate_r_squared(data, &predictions, y_mean);
    
    Ok(LinearRegressionResult {
        slope,
        intercept,
        r_squared,
        mse,
        predictions,
    })
}

/// Calcula o Erro Quadrático Médio (MSE)
pub fn calculate_mse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return 0.0;
    }
    
    let n = actual.len() as f64;
    let sum_squared_errors: f64 = actual.iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).powi(2))
        .sum();
    
    sum_squared_errors / n
}

/// Calcula o Coeficiente de Determinação (R²)
pub fn calculate_r_squared(actual: &[f64], predicted: &[f64], y_mean: f64) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return 0.0;
    }
    
    let total_sum_squares: f64 = actual.iter()
        .map(|&y| (y - y_mean).powi(2))
        .sum();
    
    let residual_sum_squares: f64 = actual.iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).powi(2))
        .sum();
    
    if total_sum_squares.abs() < f64::EPSILON {
        1.0
    } else {
        1.0 - (residual_sum_squares / total_sum_squares)
    }
}

/// Realiza previsões futuras usando os coeficientes da regressão linear
pub fn predict_future(result: &LinearRegressionResult, future_periods: usize) -> Vec<f64> {
    let n = result.predictions.len();
    (0..future_periods)
        .map(|i| result.intercept + result.slope * (n + i) as f64)
        .collect()
}

/// Calcula estatísticas descritivas básicas para uma série temporal
pub fn calculate_descriptive_stats(data: &[f64]) -> Result<(f64, f64, f64, f64), TimeSeriesError> {
    if data.is_empty() {
        return Err(TimeSeriesError::new("Dados vazios para cálculo de estatísticas"));
    }
    
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    
    let variance: f64 = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    
    let std_dev = variance.sqrt();
    let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    Ok((mean, std_dev, min, max))
}

/// Gera uma visualização ASCII art da série temporal e previsões
pub fn ascii_plot(actual: &[f64], predicted: &[f64], title: &str) {
    if actual.is_empty() || actual.len() != predicted.len() {
        println!("Dados inválidos para plotagem");
        return;
    }

    let height = 10;
    let width = actual.len() * 2;
    
    let all_values: Vec<f64> = actual.iter().chain(predicted.iter()).cloned().collect();
    let min_val = all_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = all_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;
    
    if range.abs() < f64::EPSILON {
        println!("Intervalo de dados muito pequeno para plotagem");
        return;
    }

    println!("\n{}", title);
    println!("{}", "-".repeat(width.min(60) + 12));
    
    for row in (0..height).rev() {
        let threshold = min_val + (range * (row as f64) / (height as f64));
        
        print!("{:8.1} | ", threshold);
        
        for i in 0..actual.len() {
            let is_actual = actual[i] >= threshold;
            let is_predicted = predicted[i] >= threshold;
            
            if is_actual && is_predicted {
                print!("●");
            } else if is_actual {
                print!("o");
            } else if is_predicted {
                print!("x");
            } else {
                print!(" ");
            }
            
            if i < actual.len() - 1 {
                print!(" ");
            }
        }
        println!();
    }
    
    println!("         |{}", "-".repeat(width.min(60) + 2));
    print!("          ");
    for i in 0..actual.len() {
        print!("{} ", i + 1);
        if i < actual.len() - 1 {
            print!(" ");
        }
    }
    println!("\n          Periodo");
    
    println!("\nLegenda:");
    println!("  o = Valor Real");
    println!("  x = Valor Previsto");
    println!("  ● = Real e Previsto (sobrepostos)");
}

#[cfg(test)]
mod testes {
    use super::*;

    fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
        assert!((a - b).abs() < epsilon, "{} != {} within {}", a, b, epsilon);
    }

    #[test]
    fn test_regressao_linear_ajuste_perfeito() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let result = linear_regression(&data).unwrap();
        
        assert_approx_eq(result.slope, 2.0, 1e-10);
        assert_approx_eq(result.intercept, 1.0, 1e-10);
        assert_approx_eq(result.r_squared, 1.0, 1e-10);
        assert_approx_eq(result.mse, 0.0, 1e-10);
    }

    #[test]
    fn test_regressao_linear_dados_constantes() {
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let result = linear_regression(&data).unwrap();
        
        assert_approx_eq(result.slope, 0.0, 1e-10);
        assert_approx_eq(result.intercept, 5.0, 1e-10);
        assert_approx_eq(result.r_squared, 1.0, 1e-10);
    }

    #[test]
    fn test_regressao_linear_slope_negativo() {
        let data = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let result = linear_regression(&data).unwrap();
        
        assert_approx_eq(result.slope, -1.0, 1e-10);
        assert_approx_eq(result.intercept, 5.0, 1e-10);
        assert_approx_eq(result.r_squared, 1.0, 1e-10);
    }

    #[test]
    fn test_calcular_mse() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.0, 2.0, 3.0];
        assert_approx_eq(calculate_mse(&actual, &predicted), 0.0, 1e-10);
        
        let predicted2 = vec![2.0, 3.0, 4.0];
        assert_approx_eq(calculate_mse(&actual, &predicted2), 1.0, 1e-10);
    }

    #[test]
    fn test_calcular_r_quadrado() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_mean = 3.0;
        assert_approx_eq(calculate_r_squared(&actual, &predicted, y_mean), 1.0, 1e-10);
        
        let predicted2 = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        assert_approx_eq(calculate_r_squared(&actual, &predicted2, y_mean), 0.0, 1e-10);
    }

    #[test]
    fn test_previsao_futura() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = linear_regression(&data).unwrap();
        let predictions = predict_future(&result, 3);
        
        assert_eq!(predictions.len(), 3);
        assert_approx_eq(predictions[0], 6.0, 1e-10);
        assert_approx_eq(predictions[1], 7.0, 1e-10);
        assert_approx_eq(predictions[2], 8.0, 1e-10);
    }

    #[test]
    fn test_estatisticas_descritivas() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std_dev, min, max) = calculate_descriptive_stats(&data).unwrap();
        
        assert_approx_eq(mean, 3.0, 1e-10);
        assert_approx_eq(std_dev, (2.0_f64).sqrt(), 1e-10);
        assert_approx_eq(min, 1.0, 1e-10);
        assert_approx_eq(max, 5.0, 1e-10);
    }

    #[test]
    fn test_dados_vazios() {
        let data: Vec<f64> = vec![];
        let result = linear_regression(&data);
        assert!(result.is_err());
        
        let stats_result = calculate_descriptive_stats(&data);
        assert!(stats_result.is_err());
    }

    #[test]
    fn test_unico_ponto_dado() {
        let data = vec![5.0];
        let result = linear_regression(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_caso_limite_denominador_zero() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let result = linear_regression(&data).unwrap();
        
        assert_approx_eq(result.slope, 0.0, 1e-10);
        assert_approx_eq(result.intercept, 1.0, 1e-10);
        assert_approx_eq(result.r_squared, 1.0, 1e-10);
    }

    #[test]
    fn test_documentacao_exemplos() {
        let dados = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let resultado = linear_regression(&dados).unwrap();
        
        assert!(resultado.slope.is_finite());
        assert!(resultado.intercept.is_finite());
        
        let previsoes = predict_future(&resultado, 3);
        assert_eq!(previsoes.len(), 3);
        
        ascii_plot(&dados, &resultado.predictions, "Teste");
    }

    #[test]
    fn test_ascii_plot_dados_invalidos() {
        ascii_plot(&[], &[], "Vazio");
        ascii_plot(&[1.0], &[1.0, 2.0], "Tamanhos diferentes");
    }

    #[test]
    fn test_ascii_plot_dados_constantes() {
        let data = vec![5.0, 5.0, 5.0];
        ascii_plot(&data, &data, "Constantes");
    }
}
