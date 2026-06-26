# Hero figure for the Pages poster: between- vs within-person effect of perceived
# support on next-wave depression. The story is the COLLAPSE to the null line.
suppressMessages({library(dplyr); library(geepack); library(ggplot2)})
PROC <- "data/processed"
z  <- function(x) (x - mean(x, na.rm=TRUE))/sd(x, na.rm=TRUE)
ci <- function(cf, t) exp(cf[t,"Estimate"] + c(-1.96,1.96)*cf[t,"Std.err"])
panel <- read.csv(file.path(PROC,"village_panel_tv.csv")) |>
  filter(!is.na(dep_t), !is.na(dep_prev))

fit <- function(parent){
  d <- panel |> filter(parent==!!parent) |>
    select(idnum,dep_t,dep_prev,wave_out,x=v_perceived,fh,poverty_cat,par_edu) |> na.omit()
  d$x<-z(d$x); d$poverty_cat<-z(d$poverty_cat); d$par_edu<-z(d$par_edu)
  d <- d |> group_by(idnum) |> mutate(x_ib=mean(x), x_iw=x-x_ib) |> ungroup()
  g <- geeglm(dep_t~dep_prev+factor(wave_out)+x_ib+x_iw+fh+poverty_cat+par_edu+x_iw:fh,
              id=idnum,data=d,family=binomial,corstr="exchangeable")
  cf <- summary(g)$coefficients
  bt<-ci(cf,"x_ib"); wn<-ci(cf,"x_iw")
  data.frame(parent=parent,
    term=c("Between parents","Within a parent (over time)"),
    OR=c(exp(cf["x_ib","Estimate"]),exp(cf["x_iw","Estimate"])),
    lo=c(bt[1],wn[1]), hi=c(bt[2],wn[2]))
}
df <- bind_rows(lapply(c("Mothers","Fathers"), function(p) fit(tolower(sub("s$","",p))) |> mutate(parent=p)))
df$parent <- factor(df$parent, levels=c("Mothers","Fathers"))
df$term   <- factor(df$term, levels=c("Between parents","Within a parent (over time)"))
wide <- tidyr::pivot_wider(df[c("parent","term","OR")], names_from=term, values_from=OR)

p <- ggplot(df, aes(OR, parent)) +
  geom_vline(xintercept=1, linetype=2, color="grey45") +
  annotate("text", x=1.02, y=Inf, label="no effect", hjust=0, vjust=1.6,
           size=3.1, color="grey45", fontface="italic") +
  geom_segment(data=wide, aes(x=`Between parents`, xend=`Within a parent (over time)`,
               y=parent, yend=parent), color="grey75", linewidth=1.1, inherit.aes=FALSE) +
  geom_pointrange(aes(xmin=lo, xmax=hi, color=term, shape=term), fatten=4.2, linewidth=0.9) +
  scale_x_log10(breaks=c(0.6,0.8,1.0,1.25), limits=c(0.55,1.35)) +
  scale_color_manual(values=c("Between parents"="#1b6ca8","Within a parent (over time)"="#b1402f")) +
  scale_shape_manual(values=c("Between parents"=16,"Within a parent (over time)"=21)) +
  labs(title="A parent's support network marks their depression risk — it doesn't change it",
       subtitle="Perceived instrumental support → next-wave depression. Between parents it looks protective;
within a parent, over time, it does nothing — the effect collapses to the null line.",
       x="Odds ratio for depression, log scale  (< 1 = less depression)", y=NULL, color=NULL, shape=NULL,
       caption="FFCWS, 4 waves (Year 1–Year 9), ~4,500 parents. Observational: the within-person estimate removes time-invariant\nconfounding, so this is “no within-person effect after removing stable confounding,” not a proven zero. Same pattern holds\nacross all four support facets (enacted, grandparent co-residence, religious participation) — full ablation in the report.") +
  theme_minimal(base_size=13) +
  theme(plot.title=element_text(face="bold", size=15.5),
        plot.subtitle=element_text(color="grey30", margin=margin(b=10)),
        plot.caption=element_text(color="grey45", hjust=0, size=8.3, margin=margin(t=12)),
        legend.position="top", legend.justification="left",
        panel.grid.minor=element_blank(), panel.grid.major.y=element_blank(),
        axis.text.y=element_text(face="bold", size=13))
ggsave("docs/assets/forest_between_within.png", p, width=9.7, height=4.9, dpi=150, bg="white")
cat("wrote docs/assets/forest_between_within.png\n"); print(df, digits=3)
