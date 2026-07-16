# ============================================================
# Modelli a effetti misti sui dati frase-per-frase
# Effetto fisso: corpus | Effetto casuale: video/conversazione
# ============================================================

# --- Pacchetti (installa la prima volta togliendo il commento) ---
# install.packages(c("lme4", "car", "emmeans", "performance"))
library(lme4)
library(car)          # Anova() -> test globale sull'effetto corpus
library(emmeans)      # confronti a coppie post-hoc
library(performance)  # controllo sovradispersione

# --- 1. SELEZIONA IL FILE ---
percorso <- file.choose()   # si apre una finestra per scegliere il CSV
df <- read.csv(percorso, stringsAsFactors = FALSE)

# --- Corpora da escludere ---
escludere <- c("isdt", "kiparla", "Ittok", "parlamint", "tr-it-tok", "gen","polso")
df <- df[!(df$corpus %in% escludere), ]

cat(">>> Corpora rimasti:\n")
print(table(df$corpus))

# --- 2. GRUPPO (effetto casuale) ---
# Se la colonna 'gruppo' è già stata calcolata in Python, la usiamo direttamente.
# Altrimenti la calcoliamo ora.
if (!"gruppo" %in% colnames(df)) {
  df$gruppo <- df$doc_id
  
  # Applica la regex solo se kiparla2.18 è presente tra i corpora non esclusi
  if ("kiparla2.18" %in% df$corpus) {
    kip <- df$corpus == "kiparla2.18"
    df$gruppo[kip] <- sub("_.*$", "", df$unit_id[kip])
  }
  
  df$gruppo <- paste(df$corpus, df$gruppo, sep = "::")  # id univoco tra corpora
} else {
  cat(">>> Colonna 'gruppo' rilevata dal file di Python. Uso quella preesistente.\n")
}

# --- 3. VARIABILI ---
# Forziamo l'ordine dei livelli per non avere dubbi durante la creazione dei contrasti
df$corpus <- factor(df$corpus, levels = c("kiparla2.18", "postwita", "trittok_gen", "trittok_polso"))

df$val_pos <- df$avg_gioia + df$avg_sorpresa + df$avg_fiducia + df$avg_aspettativa
df$val_neg <- df$avg_tristezza + df$avg_paura + df$avg_disgusto + df$avg_rabbia

# --- Gestione intelligente dell'offset (n_token vs clean_tokens) ---
if ("clean_tokens" %in% colnames(df)) {
  df <- df[df$clean_tokens > 0, ]
  df$log_token <- log(df$clean_tokens)
  cat(">>> Utilizzo i 'clean_tokens' (token puliti) calcolati in Python come offset.\n")
} else {
  df <- df[df$n_token > 0, ]
  df$log_token <- log(df$n_token)
  cat(">>> Utilizzo i 'n_token' (token originali) come offset.\n")
}

cat("\n>>> Unita' (gruppi) per corpus:\n")
print(tapply(df$gruppo, df$corpus, function(x) length(unique(x))))


# ==============================================================================
# --- IMPOSTAZIONE DINAMICA DEI LIMITI DI EMMEANS ---
# Calcola il limite ottimale basandosi sul numero esatto di righe nel dataset filtrato
# aggiungendo un margine di sicurezza di 100 osservazioni.
# ==============================================================================
limite_calcolato <- nrow(df) + 100
cat(paste("\n>>> Imposto dinamicamente il limite di emmeans a:", limite_calcolato, "righe.\n"))

# Sovrascrive i limiti di default senza forzare il metodo asintotico
emm_options(pbkrtest.limit = limite_calcolato, lmerTest.limit = limite_calcolato)


# ============================================================
# A) CONTEGGI -> glmer Poisson con offset log(token)
# ============================================================
conteggi <- c(Superlativi = "n_superlatives",
              `Token emotivi` = "n_emotional_tokens",
              `Lemmi emotivi` = "n_emotional_lemmas")

for (nome in names(conteggi)) {
  y <- conteggi[[nome]]
  cat("\n\n=================================================================\n")
  cat("===", nome, "(Poisson misto) ===\n")
  cat("=================================================================\n")
  
  f <- as.formula(paste0(y, " ~ corpus + offset(log_token) + (1 | gruppo)"))
  m <- tryCatch(glmer(f, data = df, family = poisson),
                error = function(e) { cat("Errore nel fit:", conditionMessage(e), "\n"); NULL })
  
  if (is.null(m)) next
  
  disp <- tryCatch(check_overdispersion(m), error = function(e) NULL)
  if (!is.null(disp)) print(disp)
  
  cat("\n-- Test globale effetto 'corpus' --\n")
  print(Anova(m, type = "II"))
  
  # Calcolo delle medie stimate
  em <- emmeans(m, ~ corpus)
  
  # --- CONTRASTI MIRATI ---
  cat("\n--> DOMANDA 1: TikTok vs. Parlato Spontaneo (KIParla)\n")
  # Confronta trittok (gen e polso) contro kiparla (livello 1)
  c_kiparla <- list(
    "trittok_gen vs KIParla"   = c(-1, 0,  1,  0),
    "trittok_polso vs KIParla" = c(-1, 0,  0,  1)
  )
  print(contrast(em, c_kiparla, adjust = "bonferroni"))
  
  cat("\n--> DOMANDA 2: TikTok vs. Scritto Social (Twitter/Postwita)\n")
  # Confronta trittok (gen e polso) contro postwita (livello 2)
  c_postwita <- list(
    "trittok_gen vs Postwita"   = c( 0, -1,  1,  0),
    "trittok_polso vs Postwita" = c( 0, -1,  0,  1)
  )
  print(contrast(em, c_postwita, adjust = "bonferroni"))
}

# ============================================================
# B) PROPORZIONI CONTINUE -> lmer
# ============================================================
continue <- c(`Valutazione positiva` = "val_pos",
              `Valutazione negativa` = "val_neg")

for (nome in names(continue)) {
  y <- continue[[nome]]
  cat("\n\n=================================================================\n")
  cat("===", nome, "(lineare misto) ===\n")
  cat("=================================================================\n")
  
  f <- as.formula(paste0(y, " ~ corpus + (1 | gruppo)"))
  m <- tryCatch(lmer(f, data = df, REML = TRUE),
                error = function(e) { cat("Errore nel fit:", conditionMessage(e), "\n"); NULL })
  if (is.null(m)) next
  
  cat("\n-- Test globale effetto 'corpus' --\n")
  print(Anova(m, type = "II"))
  
  # Calcolo delle medie stimate
  em <- emmeans(m, ~ corpus)
  
  # --- CONTRASTI MIRATI ---
  cat("\n--> DOMANDA 1: TikTok vs. Parlato Spontaneo (KIParla)\n")
  c_kiparla <- list(
    "trittok_gen vs KIParla"   = c(-1, 0,  1,  0),
    "trittok_polso vs KIParla" = c(-1, 0,  0,  1)
  )
  print(contrast(em, c_kiparla, adjust = "bonferroni"))
  
  cat("\n--> DOMANDA 2: TikTok vs. Scritto Social (Twitter/Postwita)\n")
  c_postwita <- list(
    "trittok_gen vs Postwita"   = c( 0, -1,  1,  0),
    "trittok_polso vs Postwita" = c( 0, -1,  0,  1)
  )
  print(contrast(em, c_postwita, adjust = "bonferroni"))
}

# ============================================================
# C) EMOZIONI SINGOLE -> un lmer per emozione (formato largo)
# Correzione molteplicita': Benjamini-Hochberg (FDR) GLOBALE
# su tutti i contrasti raccolti (8 emozioni x 2 confronti x 2 domande).
# Presuppone df, factor 'corpus', 'gruppo', limiti emmeans gia' definiti.
# ============================================================

emozioni <- c(`Gioia`       = "avg_gioia",
              `Sorpresa`    = "avg_sorpresa",
              `Fiducia`     = "avg_fiducia",
              `Aspettativa` = "avg_aspettativa",
              `Tristezza`   = "avg_tristezza",
              `Paura`       = "avg_paura",
              `Disgusto`    = "avg_disgusto",
              `Rabbia`      = "avg_rabbia")

# Contrasti (stesso schema degli altri blocchi)
c_kiparla <- list(
  "trittok_gen vs KIParla"   = c(-1, 0,  1,  0),
  "trittok_polso vs KIParla" = c(-1, 0,  0,  1)
)
c_postwita <- list(
  "trittok_gen vs Postwita"   = c( 0, -1,  1,  0),
  "trittok_polso vs Postwita" = c( 0, -1,  0,  1)
)

# Contenitore per raccogliere TUTTI i contrasti (p non corretti)
risultati <- list()

for (nome in names(emozioni)) {
  y <- emozioni[[nome]]
  cat("\n\n=================================================================\n")
  cat("===", nome, "(lineare misto) ===\n")
  cat("=================================================================\n")
  
  f <- as.formula(paste0(y, " ~ corpus + (1 | gruppo)"))
  m <- tryCatch(lmer(f, data = df, REML = TRUE),
                error = function(e) { cat("Errore nel fit:", conditionMessage(e), "\n"); NULL })
  if (is.null(m)) next
  
  cat("\n-- Test globale effetto 'corpus' --\n")
  print(Anova(m, type = "II"))
  
  em <- emmeans(m, ~ corpus)
  
  # --- contrasti SENZA correzione locale: la correzione e' globale, dopo ---
  d1 <- as.data.frame(contrast(em, c_kiparla,  adjust = "none"))
  d2 <- as.data.frame(contrast(em, c_postwita, adjust = "none"))
  
  d1$emozione <- nome; d1$domanda <- "D1: TikTok vs KIParla"
  d2$emozione <- nome; d2$domanda <- "D2: TikTok vs Postwita"
  
  cat("\n--> DOMANDA 1: TikTok vs. Parlato Spontaneo (KIParla) [p non corretti]\n")
  print(d1[, c("contrast", "estimate", "SE", "df", "t.ratio", "p.value")])
  cat("\n--> DOMANDA 2: TikTok vs. Scritto Social (Postwita) [p non corretti]\n")
  print(d2[, c("contrast", "estimate", "SE", "df", "t.ratio", "p.value")])
  
  risultati[[nome]] <- rbind(d1, d2)
}

# ============================================================
# CORREZIONE GLOBALE BENJAMINI-HOCHBERG (FDR)
# ============================================================
cat("\n\n#################################################################\n")
cat("### TABELLA RIASSUNTIVA + FDR (Benjamini-Hochberg) globale ###\n")
cat("#################################################################\n")

tutti <- do.call(rbind, risultati)
rownames(tutti) <- NULL

# p-value corretto FDR su TUTTI i contrasti raccolti
tutti$p_FDR <- p.adjust(tutti$p.value, method = "BH")

# ordina per significativita' e presenta le colonne utili
tutti <- tutti[order(tutti$p_FDR), ]
tutti$signif <- ifelse(tutti$p_FDR < 0.05, "*", "")

print(tutti[, c("emozione", "domanda", "contrast",
                "estimate", "SE", "t.ratio", "p.value", "p_FDR", "signif")],
      row.names = FALSE, digits = 3)

cat("\np.value = non corretto | p_FDR = Benjamini-Hochberg su", nrow(tutti), "test\n")

# (opzionale) salva su file
# write.csv(tutti, "risultati_emozioni_FDR.csv", row.names = FALSE)

# ============================================================
# PCA delle emozioni: SOLO centroidi per corpus (no nuvola di punti)
# + frecce dei loadings (quali emozioni separano i corpora)
# + ellissi di dispersione (facoltative, ~1 sd dei punti per corpus)
# Presuppone 'df' (col. avg_*, 'corpus') gia' in memoria.
# ============================================================

# install.packages("ggplot2")
library(ggplot2)

emo_cols <- c("avg_gioia", "avg_sorpresa", "avg_fiducia", "avg_aspettativa",
              "avg_tristezza", "avg_paura", "avg_disgusto", "avg_rabbia")

ok  <- complete.cases(df[, emo_cols])
X   <- as.matrix(df[ok, emo_cols])
grp <- factor(df$corpus[ok])

lev <- c("kiparla2.18"="KIParla (parlato)","postwita"="Postwita (Twitter)",
         "trittok_gen"="TrIttok_Gen","trittok_polso"="TrIttok_PolSo")
levels(grp) <- lev[levels(grp)]

Xs  <- scale(X)
pca <- prcomp(Xs, center = FALSE, scale. = FALSE)
ve  <- round(100 * summary(pca)$importance[2, 1:2], 1)

scores <- data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2], corpus = grp)

# centroidi + errore standard del baricentro (per barre d'incertezza)
agg <- do.call(rbind, lapply(split(scores, scores$corpus), function(d) {
  n <- nrow(d)
  data.frame(corpus = d$corpus[1],
             PC1 = mean(d$PC1), PC2 = mean(d$PC2),
             se1 = sd(d$PC1)/sqrt(n), se2 = sd(d$PC2)/sqrt(n))
}))

# loadings (frecce)
load <- as.data.frame(pca$rotation[, 1:2]); load$emo <- sub("^avg_","",rownames(load))
sc <- max(abs(c(agg$PC1, agg$PC2))) / max(abs(as.matrix(load[,1:2]))) * 0.9

p <- ggplot() +
  # frecce loadings
  geom_segment(data = load, aes(x=0, y=0, xend=PC1*sc, yend=PC2*sc),
               arrow = arrow(length = unit(0.15,"cm")),
               color = "grey55", linewidth = 0.4) +
  geom_text(data = load, aes(x=PC1*sc*1.12, y=PC2*sc*1.12, label=emo),
            size = 3.3, color = "grey35") +
  # barre d'incertezza del centroide (molto strette: i corpora sono grandi)
  geom_errorbar(data = agg, aes(x=PC1, ymin=PC2-se2, ymax=PC2+se2, color=corpus),
                width=0, linewidth=0.6) +
  geom_errorbarh(data = agg, aes(y=PC2, xmin=PC1-se1, xmax=PC1+se1, color=corpus),
                 height=0, linewidth=0.6) +
  # centroidi
  geom_point(data = agg, aes(PC1, PC2, fill = corpus),
             size = 6, shape = 21, color = "black", stroke = 0.9) +
  geom_text(data = agg, aes(PC1, PC2, label = corpus),
            vjust = -1.6, size = 3.4, fontface = "bold") +
  labs(title = "Baricentri dei corpora nello spazio emotivo (PCA)",
       subtitle = paste0("PC1 ", ve[1], "% | PC2 ", ve[2],
                         "% varianza | frecce = emozioni, barre = SE del baricentro"),
       x = paste0("PC1 (", ve[1], "%)"), y = paste0("PC2 (", ve[2], "%)")) +
  coord_equal() +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave("pca_centroidi.png", p, width = 8.5, height = 7, dpi = 300)
cat(">>> Salvato: pca_centroidi.png\n\n")

cat("=== Coordinate dei baricentri ===\n")
print(agg[, c("corpus","PC1","PC2")], row.names = FALSE, digits = 3)
cat("\n=== Loadings (contributo di ogni emozione a PC1/PC2) ===\n")
print(load[order(-abs(load$PC1)), c("emo","PC1","PC2")], row.names = FALSE, digits = 3)