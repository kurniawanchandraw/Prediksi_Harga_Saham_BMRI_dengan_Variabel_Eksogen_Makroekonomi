if (!require("quantmod")) {
  install.packages("quantmod")
}

library(quantmod)

symbol <- "BMRI.JK"
start_date <- as.Date("2020-01-01")
end_date <- as.Date("2025-05-01")

getSymbols(symbol, src = "yahoo", from = start_date, to = end_date)

bmri <- Ad(get(symbol))
bmri_df <- data.frame(Date = index(bmri), BMRI = coredata(bmri))

getSymbols("^JKSE", src = "yahoo", from = start_date, to = end_date)
ihsg <- Ad(JKSE)
ihsg_df <- data.frame(Date = index(ihsg), IHSG = coredata(ihsg))

merged_data <- merge(bmri_df, ihsg_df, by = "Date")
tail(merged_data)

write.csv(merged_data, file = "data/stock_BMRI_IHSG.csv", row.names = FALSE)