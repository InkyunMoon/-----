sns.scatterplot(x='ROLLING_DESCALING', y='SCALE', data=raw_data, s=300, alpha = 0.3)
sns.scatterplot(x='FUR_SZ_TEMP', y='SCALE', data=raw_data, s=300, alpha = 0.3)
sns.scatterplot(x='FUR_HZ_TEMP', y='SCALE', data=raw_data, s=300, alpha = 0.3)
sns.scatterplot(x='ROLLING_TEMP_T5', y='SCALE', data=raw_data, s=300, alpha = 0.3)

sns.scatterplot(x='ROLLING_TEMP_T5')