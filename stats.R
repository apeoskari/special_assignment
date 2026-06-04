library(tidyverse)
library(directlabels)

df <- read_csv("test_triplets_fi.csv")

p <- df |>
  filter(dist_group %in% c(0, 1, 2)) |>
  mutate(group =
           factor(
             dist_group,
             levels = c(0, 1, 2),
             labels = c("Closely related", "Moderately related", "Unrelated")
           )) |>
  ggplot(aes(
             x = primer1_to_target,
             y = primer2_to_target)) +
  # geom_point() +
  stat_density_2d(
    aes(color = group),
    geom = "contour",
    bins = 8,
    linewidth = 0.6
  ) +
  directlabels::geom_dl(
                        method = "smart.grid",
                        aes(label = group, color = group)) +
  scale_color_viridis_d(
                        option = "D",
                        guide = "none") +
  labs(
       x = "Primer 1 to target distance",
       y = "Primer 2 to target distance",
       title = "Stimuli fi (FastText) test") +
  theme_linedraw() +
  theme(
        panel.grid = element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  coord_equal(
              xlim = c(0, 1),
              ylim = c(0, 1),
              expand = FALSE)

ggsave(
  "plots/fi/stimuli_ft_fi_test.png",
  plot = p,
  height = 4,
  width = 4,
  dpi = 300,
  bg = "white"
)
