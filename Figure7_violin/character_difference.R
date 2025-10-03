# 加载必要的库
library(ggplot2)
library(readr)
library(tidyr)
library(dplyr)
library(ggpubr)   # 用于统计分析添加统计指标
library(gridExtra) # 用于组合图
library(readxl)
library(openxlsx)
library(RColorBrewer)

# 读取数据集
# data <- read_excel("C:/Users/xyy/Desktop/nodata4/AML_allnew_merged_patient_data_processed.xlsx")
data <- read_excel("../data/AML_allnew_merged_patient_data_processed.xlsx")

# 在代码开始处添加一个特征名称映射字典
feature_names_mapping <- c(
  'B' = 'CFB',
  'Admission_Value_PLT' = 'PLT at admission'
  # 如果还有其他需要修改的特征名称，可以继续添加
)

# 'A/G', 'CHE', 'AST', 'SA', 'GLU', 'CYSC', 'UBIL', 'age', 'FIB-C', 'B',
# 'Admission_Value_HGB', 'ALT', 'LDH', 'TBA', 'CK', 'DD', 'SOD', 'SAA', 'CO2', 'Pre_treatment_Mean_NEUT'
# 'GLU', 'PA', 'AST/ALT', 'GLB', 'ProBNP', 'BUN', 'AST', 'PTTA', 'ALB', 'ALP', 
# 'B', 'Pre_treatment_Mean_NEUT', 'TP', 'TBIL', 'Pre_treatment_Mean_PLT', 'CRP'
# 'SAA', 'CHE', 'DD', 'LDH1', 'A/G', 'CO2', 'PA', 'GLU', 'LDH', 'B', 'age', 'DBIL', 
# 'CKMB', 'Pre_treatment_Mean_HGB', 'CRP', 'β2-MG', 'ALB', 'TBIL', 'CK', 'BUN'
# 'CHE', 'B', 'ALB', 'PA', 'LDH1', 'SAA', 'CYSC', 'Pre_treatment_Mean_PLT', 'FIB-C', 'TP', 'age', 'pct',
# 'A/G', 'CRP', 'HBDH', 'SOD', 'DD', 'β2-MG', 'APTT', 'SA']

feature_groups <- list(
  c('SA', 'A/G', 'LDH', 'ALB', 'pct'),
  c('CHE', 'B', 'INR', 'Admission_Value_PLT', 'SAA'),
  c('GLU', 'CRP', 'β2-MG', 'TP', 'GLB'),
  c('Pre_treatment_Mean_NEUT', 'BUN', 'APTT', 'BG', 'CREA')
)

# 定义离散特征列表
categorical_columns <- c('sex', 'ALT_categorized', 'AST_categorized', 'AST/ALT', 
                         'ALP_categorized', 'DD', 'HBsAg', 'HBeAg', 'Anti-HCV', 
                         'HIV-Ab', 'Syphilis', 'BG', 'ABOZDX', 'ABOFDX', 'Rh', 'BGZGTSC')


for (group in feature_groups) {
  plot_list <- list()
  for (feature in group) {
    # 判断是否为离散特征
    is_categorical <- feature %in% categorical_columns
    
    if (is_categorical) {
      # 将数值型分类变量转换为因子
      data[[feature]] <- as.factor(data[[feature]])
      
      # 获取类别数量
      n_categories <- length(unique(data[[feature]]))
      
      # 选择配色
      colors <- brewer.pal(max(8, n_categories), "Set2")[1:n_categories]
      
      # 创建列联表
      contingency_table <- table(data[[feature]], data$bsum_cluster_label)
      
      # 进行统计检验
      # 如果期望频数小于5或2x2表，使用Fisher精确检验
      expected <- chisq.test(contingency_table)$expected
      if (any(expected < 5) || all(dim(contingency_table) == 2)) {
        test_result <- fisher.test(contingency_table, simulate.p.value = TRUE)
        test_method <- "Fisher's exact test"
      } else {
        test_result <- chisq.test(contingency_table)
        test_method <- "Chi-square test"
      }
      
      # 计算比例
      prop_data <- data %>%
        group_by(bsum_cluster_label) %>%
        count(.data[[feature]]) %>%
        group_by(bsum_cluster_label) %>%
        mutate(prop = n/sum(n))
      
      # 创建堆叠柱状图
      p <- ggplot(prop_data, aes(x = bsum_cluster_label, y = prop, 
                                 fill = .data[[feature]])) +
        geom_bar(stat = "identity", position = "stack", width = 0.7) +  # 调整柱子宽度
        scale_fill_manual(values = colors) +  # 使用选定的配色方案
        theme_bw() +
        theme(
          legend.position = "right",
          legend.title = element_text(family = "Times", size = 12),
          legend.text = element_text(family = "Times", size = 10),
          legend.key.size = unit(0.8, "cm"),  # 调整图例大小
          axis.text.x = element_text(colour = "black", family = "Times", size = 15),
          axis.text.y = element_text(family = "Times", size = 12),
          axis.title.x = element_text(family = "Times", size = 16),
          axis.title.y = element_text(family = "Times", size = 16),
          panel.border = element_blank(),
          axis.line = element_line(colour = "black", size = 1),
          panel.grid = element_blank(),
          plot.title = element_text(family = "Times", size = 14)
        ) +
        labs(
          y = "Proportion",
          x = "Cluster",
          fill = feature
        ) +
        ggtitle(paste0(
          ifelse(feature %in% names(feature_names_mapping), 
                 feature_names_mapping[feature], 
                 feature),
          "\n", 
          "\np = ", format.pval(test_result$p.value, digits = 3))) +
        scale_y_continuous(labels = scales::percent)
      
    } else {
      # 连续变量的处理保持不变
      normality_results <- data %>%
        group_by(bsum_cluster_label) %>%
        summarize(
          Normality_p_value = shapiro.test(.data[[feature]])$p.value,
          .groups = 'drop'
        )
      
      all_normal <- all(normality_results$Normality_p_value > 0.05)
      test_method <- ifelse(all_normal, "t.test", "wilcox.test")

      # #计算统计数据
      stats <- data %>%
        group_by(bsum_cluster_label) %>%
        summarise(
          Feature = feature,  # 添加特征名称
          Mean = mean(.data[[feature]], na.rm = TRUE),
          Q1 = quantile(.data[[feature]], 0.25, na.rm = TRUE),
          Median = median(.data[[feature]], na.rm = TRUE),
          Q3 = quantile(.data[[feature]], 0.75, na.rm = TRUE),
          .groups = 'drop'  # 防止dplyr 1.0.0后版本警告
        )

      print(stats) # 打印统计数据到控制台
      
      p <- ggplot(data, aes(x = bsum_cluster_label, y = .data[[feature]], 
                            fill = bsum_cluster_label)) +
        geom_violin(trim = FALSE, color = "white") +
        geom_boxplot(width = 0.1, position = position_dodge(0.9)) +
        theme_bw() +
        theme(
          legend.position = "none",
          axis.text.x = element_text(colour = "black", family = "Times", size = 18),  # 增大x轴文字
          axis.text.y = element_text(family = "Times", size = 16),     # 增大y轴文字
          axis.title.x = element_text(family = "Times", size = 20),    # 增大x轴标题
          axis.title.y = element_text(family = "Times", size = 20),    # 增大y轴标题
          panel.border = element_blank(),
          axis.line = element_line(colour = "black", size = 1),
          panel.grid = element_blank(),
          plot.title = element_text(family = "Times", size = 18, face = "bold")       # 增大标题文字
        ) +
        labs(y = "Value", x = "Cluster") +
        ggtitle(ifelse(feature %in% names(feature_names_mapping), 
                       feature_names_mapping[feature], 
                       feature)) +
        stat_compare_means(method = test_method,
                           aes(label = paste0("p = ", after_stat(p.format))),
                           label.x.npc = "center", label.y.npc = 'top',
                           size = 6)  # 增大p值文字大小
    }
    
    plot_list[[length(plot_list) + 1]] <- p
  }
  
  # 组合图表显示
  grid.arrange(grobs = plot_list, ncol = length(group))
}

# feature_groups <- list(
#   # c('CHE', 'B', 'ALB', 'PA'),
#   # c('TP', 'LDH1', 'CRP', 'Admission_Value_HGB'),
#   # c('FIB-C', 'SAA', 'SA', 'SOD'),
#   # c('GLB', 'A/G', 'LDH'),
#   # c('CYSC', 'TBA', 'TBIL', 'APTT', 'β2-MG'),
#   # c('Admission_Value_HGB'),
#   # c('CHE', 'ALB', 'PA')
#   # c('DD','DBIL','β2-MG','TBIL','CK','age','C1q')
#   # c('TP', 'GLB','β2-MG','TBIL','CYSC','TBA','APTT')
# 
#   # c('SAA', 'CHE','PTTA','LDH1'),
#   # c('A/G', 'CO2','PA', 'GLU'),
#   # c('LDH', 'B', 'ALB','CRP'),
#   # c('β2-MG', 'Pre_treatment_Mean_HGB', 'DD', 'C1q'),
#   # c('TBIL','age','DBIL', 'CK')
#   # c('Pre_treatment_Mean_HGB')
#   # c('CHE', 'A/G', 'PA')
# 
#   c('DBIL','C1q', 'BUN', 'UBIL','GLU')
# )


# # 为每个特征组生成图表
# for (group in feature_groups) {
#   # 创建空列表以保存图表
#   plot_list <- list()
#   for (feature in group) {
#     # 分别对每个类别进行正态性检验
#     normality_results <- data %>%
#       group_by(bsum_cluster_label) %>%
#       summarize(
#         Normality_p_value = shapiro.test(.data[[feature]])$p.value,
#         .groups = 'drop'
#       )
#     
#     # 判断是否所有类别都符合正态分布 正态分布用t检验，非正态分布用秩和检验
#     all_normal <- all(normality_results$Normality_p_value > 0.05)
#     test_method <- ifelse(all_normal, "t.test", "wilcox.test")
#     
#     # 计算统计数据
#     stats <- data %>%
#       group_by(bsum_cluster_label) %>%
#       summarise(
#         Feature = feature,  # 添加特征名称
#         Mean = mean(.data[[feature]], na.rm = TRUE),
#         Q1 = quantile(.data[[feature]], 0.25, na.rm = TRUE),
#         Median = median(.data[[feature]], na.rm = TRUE),
#         Q3 = quantile(.data[[feature]], 0.75, na.rm = TRUE),
#         .groups = 'drop'  # 防止dplyr 1.0.0后版本警告
#       )
#     
#     print(stats) # 打印统计数据到控制台
#     
#     p <- ggplot(data, aes(x = bsum_cluster_label, y = .data[[feature]], fill = bsum_cluster_label)) +
#       geom_violin(trim = FALSE, color = "white") +
#       geom_boxplot(width = 0.1, position = position_dodge(0.9)) +
#       theme_bw() +
#       theme(
#         legend.position = "none",
#         axis.text.x = element_text(colour = "black", family = "Times", size = 15),
#         axis.text.y = element_text(family = "Times", size = 12, face = "plain"),
#         axis.title.x = element_text(family = "Times", size = 16, face = "plain"),
#         axis.title.y = element_text(family = "Times", size = 16, face = "plain"),
#         panel.border = element_blank(),
#         axis.line = element_line(colour = "black", size = 1),
#         panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         title = element_text(family = "Times", size = 14, face = "plain")
#       ) +
#       labs(y = "Value", x = "Cluster") +
#       ggtitle(feature) +
#       stat_compare_means(method = test_method,
#                          aes(label = paste0("p = ", after_stat(p.format))),
#                          label.x.npc = "center", label.y.npc = 'top',
#                          size = 5)
#     plot_list[[length(plot_list) + 1]] <- p
#   }
#   
#   # 使用grid.arrange将当前特征组的图表组合并展示
#   grid.arrange(grobs = plot_list, ncol = length(group))
# }







# # 画离散特征的图像
# library(ggplot2)
# library(readr)
# library(dplyr)
# 
# # 读取数据集
# data <- read_csv("C:/Users/xyy/Desktop/2cluster_bsum_classification_results.csv")
# 
# # 将数值映射到血型分类
# data$ABOZDX <- factor(data$ABOZDX, levels = c(0, 1, 2, 3, 4,5), labels = c("AB","AB", "A", "B", "O","NA"))
# 
# # 汇总数据以适用于卡方检验
# table_data <- table(data$bsum_cluster_label, data$ABOZDX)
# 
# # 进行卡方检验
# chi_test <- chisq.test(table_data)
# p_value <- chi_test$p.value
# 
# # # 创建标准条形图并添加 p 值
# # bar_plot <- ggplot(data, aes(x = bsum_cluster_label, fill = ABOZDX)) +
# #   geom_bar(position = "dodge") +
# #   theme_minimal() +
# #   labs(x = "Cluster", y = "Count", title = "Distribution of Blood Types across Clusters") +
# #   scale_fill_brewer(palette = "Set1", name = "Blood Type") +
# #   theme(text = element_text(size = 12, family = "Times")) +
# #   annotate("text", x = 1, y =5 , label = sprintf("p = %.3f", p_value), hjust = 1.5)
# # # 输出图表
# # print(bar_plot)
# 
# # 创建堆叠条形图
# stacked_bar_plot <- ggplot(data, aes(x = bsum_cluster_label, fill = ABOZDX)) +
#   geom_bar(position = "fill") +
#   theme_minimal() +
#   labs(x = "Cluster", y = "Count", title = "Stacked Distribution of ABOZDX") +
#   scale_fill_brewer(palette = "Set1", name = "Blood Type") +
#   theme(text = element_text(size = 12, family = "Times")) +
#   annotate("text", x = 2, y = 1.05, label = sprintf("p = %.3f", p_value), hjust = 1.5)  # 确保此处加上 '+'
# 
# # 输出图表
# print(stacked_bar_plot)









# # 画箱线图
# 
# # Assuming 'bsum_cluster_label' is the categorical variable for comparison
# category_variable <- 'bsum_cluster_label'
# # Function to create and display boxplots in sets of four
# for (group in feature_groups) {
#   data_long <- data %>%
#     select(c(category_variable, group)) %>%
#     pivot_longer(cols = -bsum_cluster_label, names_to = "Feature", values_to = "Value")
#   
#   # Create a plot with facets for the current group of features
#   plot <- ggplot(data_long, aes(x = bsum_cluster_label, y = Value,color = bsum_cluster_label)) +
#     geom_boxplot(outlier.shape = NA, fill = NA, color = "blue") +  # Hollow boxes with specified color
#     geom_jitter(width = 0.2, size = 1, alpha = 0.5) +  # Match color to boxes
#     scale_color_manual(values = c("Severe" = "red", "Mild" = "lightblue")) +  # Custom colors
#     facet_wrap(~Feature, scales = "free_y", ncol = 2) +  # Two columns per plot
#     labs(title = "Boxplots of Various Features by Cluster Label",
#          y = "Value",
#          x = "Cluster Label") +
#     theme_minimal() +
#     theme(legend.position = "none",
#           strip.text.x = element_text(size = 8, hjust = 1),
#           axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
#           plot.title = element_text(size = 14, face = "bold"),
#           axis.title.x = element_text(size = 12, face = "bold"),
#           axis.title.y = element_text(size = 12, face = "bold"))
#   
#   print(plot)  # Display the plot
# }
