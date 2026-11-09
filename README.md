# Compte rendu – TP3 : Analyse de ventes e-commerce avec Spark SQL

**Module :** Écosystème Big Data 2 – Année universitaire 2026/2027
**Étudiant :** Salah Eddine El Assassi
**Classe :** 5IIR – Groupe G1 – Site CFC
**Environnement :** Python 3.11, PySpark 4.3.0, Java 25 (Temurin), VS Code + Jupyter

---

## Objectifs

- Charger des fichiers CSV et JSON avec un schéma explicite et auditer la qualité des données.
- Corriger des données sales : doublons, valeurs manquantes, types invalides, casse et espaces.
- Typer correctement les colonnes en tenant compte du mode ANSI de Spark 4 (`try_cast`).
- Réaliser des jointures et des agrégations en API DataFrame et en SQL.
- Utiliser une fonction de fenêtre pour imputer une valeur manquante.

---

## Partie 0 – Génération des données

Le script fourni crée le dossier `data_tp2` avec trois fichiers. La graine aléatoire fixe (`random.seed(42)`) rend le jeu de données reproductible.

```
['clients.json', 'commandes.csv', 'produits.csv']
```

| Fichier | Contenu | Défauts volontaires |
|---|---|---|
| `produits.csv` | 13 produits, 3 catégories | prix `"N/A"` pour le produit 4 |
| `clients.json` | 60 clients (JSON Lines) | 1 client sur 7 a une ville en majuscules entourée d'espaces |
| `commandes.csv` | 3000 commandes + 40 copies | quantités `"abc"` ou `-2`, dates et clients vides, statuts incohérents, doublons exacts |

---

## Partie 1 – Chargement avec schéma explicite

### Question 1.1 – SparkSession

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = (SparkSession.builder
         .appName("TP2 - Ventes")
         .master("local[*]")
         .config("spark.sql.shuffle.partitions", 8)
         .getOrCreate())
```

```
Version Spark : 4.3.0
Mode ANSI     : true
Partitions    : 8
```

`spark.sql.shuffle.partitions = 8` réduit le nombre de partitions créées après un `groupBy` ou une jointure (200 par défaut), une valeur adaptée à un petit jeu de données en local. Le **mode ANSI**, activé par défaut dans Spark 4, fait échouer toute conversion invalide au lieu de renvoyer `null`.

### Question 1.2 – Chargement des trois sources

```python
schema_produits  = "id_produit STRING, nom STRING, categorie STRING, prix STRING"
schema_commandes = ("id_commande STRING, id_client STRING, id_produit STRING, "
                    "quantite STRING, date_commande STRING, statut STRING")

produits_raw  = spark.read.option("header", True).schema(schema_produits).csv("data_tp2/produits.csv")
commandes_raw = spark.read.option("header", True).schema(schema_commandes).csv("data_tp2/commandes.csv")
clients_raw   = spark.read.json("data_tp2/clients.json")
```

| DataFrame | Nombre de lignes |
|---|---|
| `produits_raw` | 13 |
| `clients_raw` | 60 |
| `commandes_raw` | 3040 |

Toutes les colonnes des CSV sont chargées en `string`. Un typage direct (`quantite INT`) ferait échouer la lecture en mode ANSI dès la première valeur `"abc"`. Le JSON est lu avec inférence de schéma : `id_client` est détecté en `long`, mais `date_inscription` reste un `string`, car JSON n'a pas de type date.

---

## Partie 2 – Audit de la qualité des données

### Question 2.1 – Valeurs manquantes par colonne

```python
manquants = commandes_raw.select([
    F.count(F.when(F.col(c).isNull() | (F.trim(F.col(c)) == ""), 1)).alias(c)
    for c in commandes_raw.columns
])
```

```
+-----------+---------+----------+--------+-------------+------+
|id_commande|id_client|id_produit|quantite|date_commande|statut|
+-----------+---------+----------+--------+-------------+------+
|          0|        7|         0|       0|           10|     0|
+-----------+---------+----------+--------+-------------+------+
```

`F.when(condition, 1)` renvoie 1 quand la condition est vraie et `null` sinon ; `F.count` ne compte que les valeurs non nulles. La liste en compréhension produit une expression par colonne, le tout dans **un seul `select`**.

Seules `id_client` (7) et `date_commande` (10) ont des valeurs vides. `quantite` n'a aucune valeur manquante mais n'est pas propre pour autant : `"abc"` et `-2` sont des valeurs **invalides**, pas des valeurs vides (voir 2.3).

### Question 2.2 – Doublons exacts et valeurs de statut

```python
n_total = commandes_raw.count()
n_dedup = commandes_raw.dropDuplicates().count()
commandes_raw.groupBy("statut").count().orderBy(F.desc("count")).show()
```

```
Lignes totales    : 3040
Lignes distinctes : 3000
Doublons exacts   : 40

+----------+-----+
|statut    |count|
+----------+-----+
|'livree'  |1260 |
|'annulee' |476  |
|'Livree'  |465  |
|'LIVREE ' |426  |
|'en_cours'|413  |
+----------+-----+
Valeurs distinctes de statut : 5
```

**Remarque :** on compte 5 valeurs de statut pour seulement 3 statuts réels. `livree`, `Livree` et `LIVREE ` (avec un espace final) désignent le même statut. Un filtre `statut = 'livree'` sur les données brutes ignorerait 465 + 426 = **891 commandes livrées**, et le chiffre d'affaires serait fortement sous-estimé.

### Question 2.3 – Quantités non numériques ou négatives

```python
cmd_q = commandes_raw.withColumn("quantite_int", F.col("quantite").try_cast("int"))
quantites_invalides = cmd_q.filter(F.col("quantite_int").isNull() | (F.col("quantite_int") <= 0))
```

```
Lignes concernées : 35
+--------+-----+
|quantite|count|
+--------+-----+
|     abc|   20|
|      -2|   15|
+--------+-----+

+-----------+---------+----------+--------+-------------+-------+------------+
|id_commande|id_client|id_produit|quantite|date_commande| statut|quantite_int|
+-----------+---------+----------+--------+-------------+-------+------------+
|          1|       51|         8|     abc|   2025-05-05| livree|        NULL|
|          6|       35|         8|      -2|   2025-04-08|annulee|          -2|
|        151|       25|         8|     abc|   2025-09-25| livree|        NULL|
|        206|       44|         5|      -2|   2025-11-20| Livree|          -2|
|        301|       43|         5|     abc|   2025-09-27| livree|        NULL|
+-----------+---------+----------+--------+-------------+-------+------------+
```

35 lignes sont concernées : 20 quantités non numériques et 15 négatives. En mode ANSI, `cast("int")` sur `"abc"` lève une erreur `CAST_INVALID_INPUT` qui arrête le job ; `try_cast` renvoie `null` à la place, ce qui permet de détecter les valeurs invalides sans planter.

### Question 2.4 – Problème de la colonne `prix`

```python
produits_raw.filter(F.col("prix").try_cast("double").isNull()).show()
produits_raw.select("prix").distinct().orderBy("prix").show()
```

```
+----------+------+------------+----+
|id_produit|   nom|   categorie|prix|
+----------+------+------------+----+
|         4|Casque|Informatique| N/A|
+----------+------+------------+----+

+----+
|prix|
+----+
| 249|
| 399|
|  49|
|  79|
|  99|
| N/A|
+----+
```

La colonne `prix` a deux problèmes :
1. Elle est stockée en texte. Le tri le montre : `249 < 399 < 49` est un tri **caractère par caractère**, pas numérique, et aucun calcul n'est possible.
2. Le **produit 4 (Casque, catégorie Informatique)** a pour prix `"N/A"`.

Supprimer ce produit ferait perdre toutes ses commandes. Son prix sera donc remplacé par la moyenne de sa catégorie (règle R5).

---

## Partie 3 – Nettoyage et typage

Règles appliquées : **R1** doublons exacts, **R2** typage, **R3** normalisation de `statut` et `ville`, **R4** suppression des commandes invalides, **R5** imputation du prix.

### Question 3.1 – DataFrame `commandes` (R1 → R4)

```python
commandes = (commandes_raw
    .dropDuplicates()                                                       # R1
    .withColumn("id_commande",   F.col("id_commande").try_cast("int"))      # R2
    .withColumn("id_client",     F.col("id_client").try_cast("int"))
    .withColumn("id_produit",    F.col("id_produit").try_cast("int"))
    .withColumn("quantite",      F.col("quantite").try_cast("int"))
    .withColumn("date_commande", F.col("date_commande").try_cast("date"))
    .withColumn("statut",        F.lower(F.trim(F.col("statut"))))         # R3
    .filter(F.col("quantite").isNotNull() & (F.col("quantite") > 0))       # R4
    .filter(F.col("date_commande").isNotNull())
    .filter(F.col("id_client").isNotNull())
)
```

```
Avant nettoyage : 3040 lignes
Après nettoyage : 2949 lignes (91 supprimées)

root
 |-- id_commande: integer (nullable = true)
 |-- id_client: integer (nullable = true)
 |-- id_produit: integer (nullable = true)
 |-- quantite: integer (nullable = true)
 |-- date_commande: date (nullable = true)
 |-- statut: string (nullable = true)

+--------+-----+
|  statut|count|
+--------+-----+
|  livree| 2089|
| annulee|  463|
|en_cours|  397|
+--------+-----+
```

On passe de 3040 à 2949 lignes : 40 doublons supprimés par R1, puis 51 lignes invalides supprimées par R4. Le total (51) est inférieur à 35 + 10 + 7 = 52 parce qu'une même ligne peut cumuler plusieurs défauts et que certains défauts se trouvaient dans des doublons déjà retirés. Le filtre R4 est appliqué **après** le typage : toutes les valeurs invalides étant devenues `null` avec `try_cast`, un simple `isNotNull()` suffit. Il ne reste plus que 3 statuts.

### Question 3.2 – DataFrame `produits` (R2 + R5)

```python
w = Window.partitionBy("categorie")

produits = (produits_raw
    .withColumn("id_produit", F.col("id_produit").try_cast("int"))
    .withColumn("prix",       F.col("prix").try_cast("double"))
    .withColumn("prix", F.round(F.coalesce(F.col("prix"), F.avg("prix").over(w)), 2))
    .orderBy("id_produit"))
```

```
+----------+-------------+------------+-----+
|id_produit|          nom|   categorie| prix|
+----------+-------------+------------+-----+
|         1|      Clavier|Informatique|399.0|
|         2|       Souris|Informatique| 49.0|
|         3|        Ecran|Informatique| 49.0|
|         4|       Casque|Informatique|149.0|
|         5|       Webcam|Informatique| 99.0|
|         6|        Lampe|      Maison| 79.0|
|         7|       Chaise|      Maison| 79.0|
|         8|       Bureau|      Maison| 79.0|
|         9|      Etagere|      Maison|399.0|
|        10|       Ballon|       Sport| 49.0|
|        11|     Raquette|       Sport|399.0|
|        12|Tapis de yoga|       Sport|399.0|
|        13|     Haltères|       Sport|249.0|
+----------+-------------+------------+-----+
```

Vérification : la moyenne Informatique hors Casque vaut (399 + 49 + 49 + 99) / 4 = **149.0**.

On utilise une **fonction de fenêtre** plutôt qu'un `groupBy` : `Window.partitionBy("categorie")` calcule la moyenne par catégorie tout en conservant les 13 lignes. `avg` ignore les `null`, donc la valeur manquante ne fausse pas le calcul, et `coalesce` ne remplace que le prix manquant.

### Question 3.3 – DataFrame `clients` (R2 + R3)

```python
clients = (clients_raw
    .withColumn("id_client",        F.col("id_client").cast("int"))
    .withColumn("date_inscription", F.col("date_inscription").try_cast("date"))
    .withColumn("ville",            F.initcap(F.trim(F.col("ville")))))
```

```
root
 |-- date_inscription: date (nullable = true)
 |-- id_client: integer (nullable = true)
 |-- nom: string (nullable = true)
 |-- ville: string (nullable = true)

Nombre de villes distinctes : 6
+----------+
|     ville|
+----------+
|    Agadir|
|Casablanca|
|       Fès|
| Marrakech|
|     Rabat|
|    Tanger|
+----------+
```

Avant nettoyage, la colonne contenait des variantes comme `" RABAT "` ou `" CASABLANCA "`. Après `trim` puis `initcap`, il reste exactement **6 villes cohérentes**. L'ordre compte : sans `trim` d'abord, le premier caractère serait un espace et la vraie première lettre ne serait pas mise en majuscule. `initcap` gère les accents (`" FÈS "` donne `"Fès"`).

---

## Partie 4 – Requêtes SQL sur vues temporaires

Le chiffre d'affaires (CA) ne compte que les commandes de statut `livree` et vaut `quantite × prix`.

### Question 4.1 – Vues temporaires et CA total

```python
produits.createOrReplaceTempView("produits")
clients.createOrReplaceTempView("clients")
commandes.createOrReplaceTempView("commandes")
```

```sql
SELECT ROUND(SUM(c.quantite * p.prix), 2) AS ca_total,
       COUNT(*)                           AS nb_commandes_livrees
FROM commandes c
JOIN produits p ON c.id_produit = p.id_produit
WHERE c.statut = 'livree'
```

```
+---------+--------------------+
| ca_total|nb_commandes_livrees|
+---------+--------------------+
|1215123.0|                2089|
+---------+--------------------+
```

Le CA total s'élève à **1 215 123** pour **2089 commandes livrées**. Ce résultat dépend directement du nettoyage : la normalisation des statuts (R3) réintègre 891 commandes, et l'imputation du prix du Casque (R5) évite qu'un prix `null` annule le montant de ses commandes.

### Question 4.2 – CA par catégorie

```sql
SELECT p.categorie,
       ROUND(SUM(c.quantite * p.prix), 2)            AS ca,
       COUNT(*)                                      AS nb_commandes,
       ROUND(SUM(c.quantite * p.prix) / COUNT(*), 2) AS panier_moyen
FROM commandes c
JOIN produits p ON c.id_produit = p.id_produit
WHERE c.statut = 'livree'
GROUP BY p.categorie
ORDER BY ca DESC
```

```
+------------+--------+------------+------------+
|   categorie|      ca|nb_commandes|panier_moyen|
+------------+--------+------------+------------+
|       Sport|574687.0|         685|      838.96|
|Informatique|354311.0|         800|      442.89|
|      Maison|286125.0|         604|      473.72|
+------------+--------+------------+------------+
```

**Sport** est la catégorie la plus rentable (environ 47 % du CA) alors qu'elle a moins de commandes qu'Informatique (685 contre 800). Son panier moyen élevé (838.96) s'explique par trois produits chers (Raquette et Tapis de yoga à 399, Haltères à 249). Informatique a le plus de commandes mais le panier moyen le plus faible, à cause de la Souris et de l'Écran à 49.

### Question 4.3 – CA mensuel

```sql
SELECT date_format(c.date_commande, 'yyyy-MM') AS mois,
       ROUND(SUM(c.quantite * p.prix), 2)      AS ca,
       COUNT(*)                                AS nb_commandes
FROM commandes c
JOIN produits p ON c.id_produit = p.id_produit
WHERE c.statut = 'livree'
GROUP BY date_format(c.date_commande, 'yyyy-MM')
ORDER BY mois
```

```
+-------+--------+------------+
|   mois|      ca|nb_commandes|
+-------+--------+------------+
|2025-01|118017.0|         183|
|2025-02| 98790.0|         163|
|2025-03|109451.0|         189|
|2025-04| 81453.0|         145|
|2025-05| 86787.0|         158|
|2025-06|114252.0|         186|
|2025-07|106081.0|         193|
|2025-08|114002.0|         186|
|2025-09| 92077.0|         155|
|2025-10| 90417.0|         164|
|2025-11| 99390.0|         173|
|2025-12|104406.0|         194|
+-------+--------+------------+

Mois au CA le plus élevé : 2025-01 avec un CA de 118,017
```

Le mois au CA le plus élevé est **janvier 2025 (118 017)** et le plus faible **avril 2025 (81 453)**. Janvier n'a pourtant pas le plus grand nombre de commandes (décembre : 194, juillet : 193) : le CA dépend aussi des produits commandés et des quantités. Les données étant générées aléatoirement, ces écarts ne traduisent pas une vraie saisonnalité.

### Question 4.4 – Top 5 des produits par CA

```sql
SELECT p.nom,
       p.categorie,
       SUM(c.quantite)                    AS quantite_totale,
       ROUND(SUM(c.quantite * p.prix), 2) AS ca
FROM commandes c
JOIN produits p ON c.id_produit = p.id_produit
WHERE c.statut = 'livree'
GROUP BY p.nom, p.categorie
ORDER BY ca DESC
LIMIT 5
```

```
+-------------+------------+---------------+--------+
|          nom|   categorie|quantite_totale|      ca|
+-------------+------------+---------------+--------+
|     Raquette|       Sport|            552|220248.0|
|Tapis de yoga|       Sport|            517|206283.0|
|      Clavier|Informatique|            491|195909.0|
|      Etagere|      Maison|            451|179949.0|
|     Haltères|       Sport|            485|120765.0|
+-------------+------------+---------------+--------+
```

Les 4 premiers produits sont tous au prix le plus élevé du catalogue (399), et 3 des 5 produits du top appartiennent à la catégorie Sport, ce qui confirme la question 4.2. Les Haltères vendent plus d'unités que l'Étagère (485 contre 451) mais génèrent un CA plus faible : à volume comparable, c'est le prix unitaire qui détermine le classement.

### Vérification croisée – API DataFrame

```python
(commandes.filter(F.col("statut") == "livree")
 .join(produits, "id_produit")
 .agg(F.round(F.sum(F.col("quantite") * F.col("prix")), 2).alias("ca_total"),
      F.count("*").alias("nb_commandes_livrees")))
```

Le résultat (1215123.0 et 2089) est identique à celui de la requête SQL de la question 4.1. SQL et API DataFrame passent par le même optimiseur (Catalyst) et produisent le même résultat.

---

## Synthèse des résultats

| Indicateur | Valeur |
|---|---|
| Commandes brutes / après nettoyage | 3040 / 2949 |
| Doublons exacts | 40 |
| Quantités invalides | 35 (20 `abc`, 15 négatives) |
| Valeurs manquantes | 7 clients, 10 dates |
| Prix invalide | produit 4 (Casque), imputé à 149.0 |
| CA total (commandes livrées) | 1 215 123 (2089 commandes) |
| Catégorie la plus rentable | Sport (574 687) |
| Mois au CA le plus élevé | 2025-01 (118 017) |
| Produit au CA le plus élevé | Raquette (220 248) |

---

## Conclusion

Ce TP montre qu'il faut **auditer avant d'analyser**. Charger toutes les colonnes en `string`, puis les convertir avec `try_cast`, permet de traiter des données sales sans faire planter Spark en mode ANSI. L'audit a révélé des doublons, des valeurs manquantes, des types invalides et des incohérences de casse et d'espaces. Sans nettoyage, le chiffre d'affaires aurait été faux : 891 commandes livrées auraient été ignorées à cause de la casse du statut, et les commandes du Casque auraient eu un prix `null`. Enfin, les vues temporaires permettent d'exprimer les analyses en SQL avec le même résultat qu'en API DataFrame.
