use timeseries_analysis::*;

fn main() {
    // Exemplo de uso da biblioteca
    let sales_data = vec![100.0, 120.0, 130.0, 145.0, 160.0];
    
    println!("=== ANALISE DE SERIES TEMPORAIS ===");
    println!("Dados: {:?}", sales_data);
    
    // Calcular estatisticas descritivas
    if let Ok((mean, std_dev, min, max)) = calculate_descriptive_stats(&sales_data) {
        println!("\nEstatisticas Descritivas:");
        println!("   Media: {:.2}", mean);
        println!("   Desvio Padrao: {:.2}", std_dev);
        println!("   Minimo: {:.2}", min);
        println!("   Maximo: {:.2}", max);
    }
    
    // Realizar regressao linear
    match linear_regression(&sales_data) {
        Ok(result) => {
            println!("\nResultado da Regressao Linear:");
            println!("   Slope (β1): {:.4}", result.slope);
            println!("   Intercept (β0): {:.4}", result.intercept);
            println!("   R²: {:.4}", result.r_squared);
            println!("   MSE: {:.4}", result.mse);
            
            println!("\nValores Previstos:");
            for (i, (actual, predicted)) in sales_data.iter().zip(result.predictions.iter()).enumerate() {
                println!("   Periodo {}: Real = {:.1}, Previsto = {:.1}", i + 1, actual, predicted);
            }
            
            // Gerar gráfico ASCII
            ascii_plot(&sales_data, &result.predictions, "Vendas - Real vs Previsto");
            
            // Fazer previsoes futuras
            let forecasts = predict_future(&result, 3);
            println!("\nPrevisoes para os proximos 3 periodos:");
            for (i, forecast) in forecasts.iter().enumerate() {
                println!("   Periodo {}: {:.2}", i + 6, forecast); // Começa do período 6
            }

            // Gerar gráfico com previsões futuras
            let all_data: Vec<f64> = sales_data.iter().chain(forecasts.iter()).cloned().collect();
            let all_predictions: Vec<f64> = result.predictions.iter().chain(forecasts.iter()).cloned().collect();
            ascii_plot(&all_data, &all_predictions, "Vendas - Serie Completa com Previsoes");
        }
        Err(e) => println!("\nErro na regressao: {}", e),
    }

    // Exemplo adicional com dados perfeitos
    println!("\n\n=== EXEMPLO ADICIONAL: DADOS PERFEITOS ===");
    let perfect_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    if let Ok(perfect_result) = linear_regression(&perfect_data) {
        ascii_plot(&perfect_data, &perfect_result.predictions, "Dados Perfeitos - y = x + 0");
    }
}
