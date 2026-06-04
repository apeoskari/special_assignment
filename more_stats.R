library(tidyverse)
library(knitr)

df <- read_csv("triplets_sv.csv")

stats <- df |>
  filter(dist_group %in% c(0, 1, 2)) |>
  mutate(
    group = factor(
      dist_group,
      levels = c(0, 1, 2),
      labels = c("Closely related", "Moderately related", "Unrelated")
    )
  ) |>
  group_by(group) |>
  summarise(
    n = n(),
    primer1_median = median(primer1_to_target, na.rm = TRUE),
    primer1_sd     = sd(primer1_to_target, na.rm = TRUE),
    primer2_median = median(primer2_to_target, na.rm = TRUE),
    primer2_sd     = sd(primer2_to_target, na.rm = TRUE),
    .groups = "drop"
  ) |>
  # Kosmeettista muotoilua (pyöristys)
  mutate(
    across(
      c(primer1_median, primer1_sd, primer2_median, primer2_sd),
      ~ round(.x, 3)
    )
  )

# Tulosta siistinä konsoliin
stats |>
  rename(
    `Group` = group,
    `n` = n,
    `Primer 1 median` = primer1_median,
    `Primer 1 SD`     = primer1_sd,
    `Primer 2 median` = primer2_median,
    `Primer 2 SD`     = primer2_sd
  ) |>
  kable(
    caption = "FastText, stimuli SV",
    align = "lrrrrr",
    booktabs = TRUE
  )

# Tallenna myös CSV:nä
write_csv(stats, "plots/sv/stimuli_ft_sv_summary.csv")
