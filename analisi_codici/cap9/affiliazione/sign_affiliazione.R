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
# (Modifica questa lista in base ai test che vuoi effettuare)
escludere <- c("isdt", "kiparla", "Ittok", "parlamint", "tr-it-tok", "gen","polso")
# Puoi aggiungere alla riga sopra anche "gen", "polso", "trittok_gen", "trittok_polso" o "kiparla2.18" all'occorrenza.

df <- df[!(df$corpus %in% escludere), ]

cat(">>> Corpora rimasti:\n")
print(table(df$corpus))

# --- 2. GRUPPO (effetto casuale) ---
# video per i social; per KIParla la conversazione (prefisso di unit_id prima del primo "_")
df$gruppo <- df$doc_id

# Applica la regex solo se kiparla2.18 è presente tra i corpora non esclusi
if ("kiparla2.18" %in% df$corpus) {
  kip <- df$corpus == "kiparla2.18"
  df$gruppo[kip] <- sub("_.*$", "", df$unit_id[kip])
}

df$gruppo <- paste(df$corpus, df$gruppo, sep = "::")  # id univoco tra corpora

# --- 3. VARIABILI ---
# Imposta KIParla come livello di riferimento solo se è presente
if ("kiparla2.18" %in% unique(df$corpus)) {
  df$corpus <- relevel(factor(df$corpus), ref = "kiparla2.18")
} else {
  df$corpus <- factor(df$corpus)
}

df$val_pos <- df$avg_gioia + df$avg_sorpresa + df$avg_fiducia + df$avg_aspettativa
df$val_neg <- df$avg_tristezza + df$avg_paura + df$avg_disgusto + df$avg_rabbia

df <- df[df$n_token > 0, ]
df$log_token <- log(df$n_token)

cat("\n>>> Unita' (gruppi) per corpus:\n")
print(tapply(df$gruppo, df$corpus, function(x) length(unique(x))))

# ============================================================
# A) CONTEGGI -> glmer Poisson con offset log(token)
# ============================================================
conteggi <- c(Superlativi = "n_superlatives",
              `Token emotivi` = "n_emotional_tokens",
              `Lemmi emotivi` = "n_emotional_lemmas")

for (nome in names(conteggi)) {
  y <- conteggi[[nome]]
  cat("\n\n=========================================\n")
  cat("===", nome, "(Poisson misto) ===\n")
  cat("=========================================\n")
  
  f <- as.formula(paste0(y, " ~ corpus + offset(log_token) + (1 | gruppo)"))
  m <- tryCatch(glmer(f, data = df, family = poisson),
                error = function(e) { cat("Errore nel fit:", conditionMessage(e), "\n"); NULL })
  
  if (is.null(m)) next
  
  disp <- tryCatch(check_overdispersion(m), error = function(e) NULL)
  if (!is.null(disp)) print(disp)
  
  cat("\n-- Test globale effetto 'corpus' --\n")
  print(Anova(m, type = "II"))
  
  cat("\n-- Confronti a coppie (emmeans, Tukey) --\n")
  print(pairs(emmeans(m, ~ corpus)))
}

# ============================================================
# B) PROPORZIONI CONTINUE -> lmer
# ============================================================
continue <- c(`Valutazione positiva` = "val_pos",
              `Valutazione negativa` = "val_neg")

for (nome in names(continue)) {
  y <- continue[[nome]]
  cat("\n\n=========================================\n")
  cat("===", nome, "(lineare misto) ===\n")
  cat("=========================================\n")
  
  f <- as.formula(paste0(y, " ~ corpus + (1 | gruppo)"))
  m <- tryCatch(lmer(f, data = df, REML = TRUE),
                error = function(e) { cat("Errore nel fit:", conditionMessage(e), "\n"); NULL })
  if (is.null(m)) next
  
  cat("\n-- Test globale effetto 'corpus' --\n")
  print(Anova(m, type = "II"))
  
  cat("\n-- Confronti a coppie (emmeans, Tukey) --\n")
  print(pairs(emmeans(m, ~ corpus)))
}