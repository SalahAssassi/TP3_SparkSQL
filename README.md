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
- Utiliser des fonctions de fenêtre (classement, cumul, `lag`) et des CTE en SQL.
- Lire les plans d'exécution (`explain`) : predicate pushdown, broadcast, cache.
- Écrire et relire des données au format Parquet partitionné, et produire un rapport CSV.

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

## Partie 5 – Jointures

### Question 5.1 – DataFrame `ventes` (commandes livrées + produits + clients)

```python
ventes = (commandes
    .filter(F.col("statut") == "livree")
    .join(produits.withColumnRenamed("nom", "produit"), "id_produit")   # inner par défaut
    .join(clients.withColumnRenamed("nom", "nom_client"), "id_client")
    .select("id_commande", "date_commande", "nom_client", "ville", "id_client",
            F.col("produit").alias("nom"), "categorie",
            "quantite", "prix",
            (F.col("quantite") * F.col("prix")).alias("ca"))
)

print("Nombre de ventes :", ventes.count())
ventes.printSchema()
ventes.show(5)
```

```
Nombre de ventes : 2089
root
 |-- id_commande: integer (nullable = true)
 |-- date_commande: date (nullable = true)
 |-- nom_client: string (nullable = true)
 |-- ville: string (nullable = true)
 |-- id_client: integer (nullable = true)
 |-- nom: string (nullable = true)
 |-- categorie: string (nullable = true)
 |-- quantite: integer (nullable = true)
 |-- prix: double (nullable = true)
 |-- ca: double (nullable = true)

+-----------+-------------+------------------+----------+---------+-------------+------------+--------+-----+------+
|id_commande|date_commande|        nom_client|     ville|id_client|          nom|   categorie|quantite| prix|    ca|
+-----------+-------------+------------------+----------+---------+-------------+------------+--------+-----+------+
|          7|   2025-04-04|       Salma Fassi|     Rabat|       18|       Bureau|      Maison|       2| 79.0| 158.0|
|         14|   2025-09-07|   Anas El Idrissi|Casablanca|       10|       Casque|Informatique|       3|149.0| 447.0|
|         15|   2025-01-30|       Imane Darif| Marrakech|       38|Tapis de yoga|       Sport|       5|399.0|1995.0|
|         24|   2025-12-10|        Hamza Tazi|     Rabat|       42|       Webcam|Informatique|       4| 99.0| 396.0|
|         30|   2025-12-01|Khadija El Idrissi| Marrakech|       34|      Clavier|Informatique|       5|399.0|1995.0|
+-----------+-------------+------------------+----------+---------+-------------+------------+--------+-----+------+
only showing top 5 rows
```

Le DataFrame `ventes` compte **2089 lignes**, soit exactement le nombre de commandes livrées de la question 4.1. Les colonnes `quantite`, `prix` et `ca = quantite × prix` sont typées (entier, double). La jointure est un `inner` (valeur par défaut) ; un `left` donne le même résultat ici (2089 lignes), car toutes les commandes livrées ont un produit et un client existants (voir la question 5.4).

### Question 5.2 – CA par ville

```python
ca_ville = (ventes
    .groupBy("ville")
    .agg(F.round(F.sum("ca"), 2).alias("ca"),
         F.count("*").alias("nb_commandes"))
    .orderBy(F.desc("ca"))
)
ca_ville.show()
```

```python
somme_villes = ca_ville.agg(F.sum("ca")).first()[0]
ca_total_41 = spark.sql("""
    SELECT SUM(c.quantite * p.prix)
    FROM commandes c JOIN produits p ON c.id_produit = p.id_produit
    WHERE c.statut = 'livree'
""").first()[0]

print(f"Somme des CA par ville : {somme_villes:,.2f}")
print(f"CA total (question 4.1) : {ca_total_41:,.2f}")
print("Cohérent :", round(somme_villes, 2) == round(ca_total_41, 2))
```

```
+----------+--------+------------+
|     ville|      ca|nb_commandes|
+----------+--------+------------+
|     Rabat|253369.0|         412|
|    Tanger|220226.0|         361|
|       Fès|209395.0|         352|
| Marrakech|208387.0|         383|
|    Agadir|170305.0|         290|
|Casablanca|153441.0|         291|
+----------+--------+------------+

Somme des CA par ville : 1,215,123.00
CA total (question 4.1) : 1,215,123.00
Cohérent : True
```

**Rabat** réalise le CA le plus élevé (253 369, pour 412 commandes) et **Casablanca** le plus faible (153 441, pour 291 commandes). La somme des CA par ville (1 215 123) est identique au CA total de la question 4.1 : aucune vente n'a été perdue ni dupliquée par les jointures.

### Question 5.3 – Clients sans commande (`left_anti`)

```python
clients_sans_commande = clients.join(commandes, "id_client", "left_anti")

print("Clients sans commande :", clients_sans_commande.count())
clients_sans_commande.orderBy("id_client").show()
```

```
Clients sans commande : 6
+---------+----------------+------------------+----------+
|id_client|date_inscription|               nom|     ville|
+---------+----------------+------------------+----------+
|       55|      2024-04-22|   Omar El Idrissi|    Tanger|
|       56|      2024-11-19|     Youssef Fassi|Casablanca|
|       57|      2024-01-17|       Omar Alaoui|Casablanca|
|       58|      2024-05-01|Youssef El Idrissi|       Fès|
|       59|      2024-10-03|       Anas Benani|       Fès|
|       60|      2024-08-30|  Ayoub El Idrissi|     Rabat|
+---------+----------------+------------------+----------+
```

**6 clients** (id 55 à 60) n'ont passé aucune commande. Un `left_anti` conserve les lignes de gauche sans correspondance à droite, sans ajouter de colonnes de `commandes`. Les 54 autres clients ont au moins une commande.

### Question 5.4 – Comparaison des types de jointure

```python
types = ["inner", "left", "left_semi", "left_anti"]
for t in types:
    res = clients.join(commandes, "id_client", t)
    print(f"{t:10s} -> {res.count():5d} lignes | {len(res.columns):2d} colonnes")

# Vérification pour la question 5.1 : inner vs left
livrees = commandes.filter(F.col("statut") == "livree")
n_inner = livrees.join(produits, "id_produit").join(clients, "id_client").count()
n_left  = livrees.join(produits, "id_produit", "left").join(clients, "id_client", "left").count()
print(f"\n5.1 en inner : {n_inner} | 5.1 en left : {n_left}")
```

```
inner      ->  2949 lignes |  9 colonnes
left       ->  2955 lignes |  9 colonnes
left_semi  ->    54 lignes |  4 colonnes
left_anti  ->     6 lignes |  4 colonnes

5.1 en inner : 2089 | 5.1 en left : 2089
```

inner : ne garde que les lignes qui ont une correspondance des deux côtés. Les colonnes des deux tables sont conservées.

left (left outer) : garde toutes les lignes de la table de gauche. Quand il n’y a pas de correspondance à droite, les colonnes de droite valent null.

left_semi : garde les lignes de gauche qui ont une correspondance à droite, mais sans ajouter de colonnes de la table de droite et sans dupliquer les lignes. C’est l’équivalent d’un WHERE EXISTS en SQL.

left_anti : garde les lignes de gauche qui n’ont aucune correspondance à droite, sans colonnes de droite. C’est l’équivalent d’un WHERE NOT EXISTS. left_semi et left_anti sont complémentaires : 54 + 6 = 60 clients.

Pour la question 5.1, un left join donnerait le même résultat qu’un inner (2089 lignes), car toutes les commandes livrées ont un produit (id_produit de 1 à 13) et un client (id_client de 1 à 54) qui existent dans les référentiels. Les résultats seraient différents si une commande faisait référence à un produit ou un client absent de sa table, par exemple un produit supprimé du catalogue ou un id_client erroné. Le inner supprimerait alors silencieusement cette commande et son CA. Le left la conserverait, avec prix, categorie ou nom_client à null, et son ca vaudrait aussi null.

---

## Partie 6 – Fonctions de fenêtre et agrégations

### Question 6.1 – Top 2 des produits par catégorie

```python
# 1) CA par produit (une ligne par produit)
ca_produit = (ventes
    .groupBy("categorie", "nom")
    .agg(F.round(F.sum("ca"), 2).alias("ca"))
)

# 2) Fenêtre : un classement séparé par catégorie, du plus gros CA au plus petit
w_rang = Window.partitionBy("categorie").orderBy(F.desc("ca"))

top2 = (ca_produit
    .withColumn("rang", F.dense_rank().over(w_rang))
    .filter(F.col("rang") <= 2)
    .orderBy("categorie", "rang")
)
top2.show()
```

```
+------------+-------------+--------+----+
|   categorie|          nom|      ca|rang|
+------------+-------------+--------+----+
|Informatique|      Clavier|195909.0|   1|
|Informatique|       Casque| 60941.0|   2|
|      Maison|      Etagere|179949.0|   1|
|      Maison|       Bureau| 39184.0|   2|
|       Sport|     Raquette|220248.0|   1|
|       Sport|Tapis de yoga|206283.0|   2|
+------------+-------------+--------+----+
```

Dans chaque catégorie, le produit n°1 est très largement devant le n°2 en Informatique (Clavier 195 909 contre Casque 60 941) et en Maison (Etagere 179 949 contre Bureau 39 184), parce que ce sont les seuls produits à 399 de leur catégorie. En Sport, les deux premiers sont proches (Raquette et Tapis de yoga, tous deux à 399). On remarque que le Casque, dont le prix avait été imputé à 149 (R5), arrive 2e de sa catégorie : sans cette imputation, son CA aurait été null et il aurait disparu du classement.

### Question 6.2 – CA cumulé et variation mensuelle

```python
# 1) CA par mois (une ligne par mois)
ca_mois = (ventes
    .groupBy(F.date_format("date_commande", "yyyy-MM").alias("mois"))
    .agg(F.round(F.sum("ca"), 2).alias("ca"))
)

# 2) Fenêtre ordonnée par mois (pas de partition : toute l'année est une seule série)
w_mois = Window.orderBy("mois")

evolution = (ca_mois
    .withColumn("ca_cumule",
        F.sum("ca").over(w_mois.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
    .withColumn("ca_mois_precedent", F.lag("ca", 1).over(w_mois))
    .withColumn("variation_pct",
        F.round((F.col("ca") - F.col("ca_mois_precedent")) / F.col("ca_mois_precedent") * 100, 2))
    .orderBy("mois")
)
evolution.show()
```

```
+-------+--------+---------+-----------------+-------------+
|   mois|      ca|ca_cumule|ca_mois_precedent|variation_pct|
+-------+--------+---------+-----------------+-------------+
|2025-01|118017.0| 118017.0|             NULL|         NULL|
|2025-02| 98790.0| 216807.0|         118017.0|       -16.29|
|2025-03|109451.0| 326258.0|          98790.0|        10.79|
|2025-04| 81453.0| 407711.0|         109451.0|       -25.58|
|2025-05| 86787.0| 494498.0|          81453.0|         6.55|
|2025-06|114252.0| 608750.0|          86787.0|        31.65|
|2025-07|106081.0| 714831.0|         114252.0|        -7.15|
|2025-08|114002.0| 828833.0|         106081.0|         7.47|
|2025-09| 92077.0| 920910.0|         114002.0|       -19.23|
|2025-10| 90417.0|1011327.0|          92077.0|         -1.8|
|2025-11| 99390.0|1110717.0|          90417.0|         9.92|
|2025-12|104406.0|1215123.0|          99390.0|         5.05|
+-------+--------+---------+-----------------+-------------+
```

Le CA cumulé atteint 1 215 123 en décembre, exactement le CA total de la question 4.1, ce qui valide le calcul du cumul. La plus forte hausse a lieu en juin (+31,65 %) et la plus forte baisse en avril (−25,58 %). La variation de janvier est null car lag n’a pas de mois précédent. Sans partitionBy, la fenêtre regroupe toutes les lignes dans une seule partition : acceptable pour 12 mois, à éviter sur de gros volumes.

### Question 6.3 – Segmentation des clients (Gold / Silver / Bronze)

```python
ca_client = (ventes
    .groupBy("id_client", "nom_client", "ville")
    .agg(F.round(F.sum("ca"), 2).alias("ca_total"))
)

segments = ca_client.withColumn("segment",
    F.when(F.col("ca_total") >= 25000, "Gold")
     .when(F.col("ca_total") >= 20000, "Silver")
     .otherwise("Bronze")
)

segments.orderBy(F.desc("ca_total")).show(10)

print("Nombre de clients par segment :")
segments.groupBy("segment").count().orderBy(F.desc("count")).show()
```

```
+---------+-----------------+---------+--------+-------+
|id_client|       nom_client|    ville|ca_total|segment|
+---------+-----------------+---------+--------+-------+
|       37|     Hamza Alaoui|      Fès| 41736.0|   Gold|
|       41|Meryem El Idrissi|      Fès| 38512.0|   Gold|
|        6|Khadija Cherkaoui|    Rabat| 35376.0|   Gold|
|       43|    Meryem Alaoui|   Tanger| 32523.0|   Gold|
|       39| Salma El Idrissi|   Agadir| 31485.0|   Gold|
|       18|      Salma Fassi|    Rabat| 30804.0|   Gold|
|       52|      Hamza Darif|Marrakech| 29435.0|   Gold|
|        9| Hamza El Idrissi|      Fès| 29401.0|   Gold|
|       50|      Ayoub Darif|   Tanger| 29270.0|   Gold|
|       38|      Imane Darif|Marrakech| 28751.0|   Gold|
+---------+-----------------+---------+--------+-------+
only showing top 10 rows
Nombre de clients par segment :
+-------+-----+
|segment|count|
+-------+-----+
| Bronze|   22|
|   Gold|   16|
| Silver|   16|
+-------+-----+
```

Sur les 54 clients actifs, on compte 16 Gold, 16 Silver et 22 Bronze. Le meilleur client est Hamza Alaoui (Fès), avec 41 736 de CA. Les 6 clients sans commande (question 5.3) n’apparaissent dans aucun segment, puisqu’ils n’ont pas de CA. Le regroupement se fait sur id_client et non sur le nom : un regroupement par nom fusionnerait des homonymes et donnerait seulement 37 « clients ».

---

## Partie 7 – Requêtes SQL avancées (CTE et `CASE WHEN`)

### Question 7.1 – Répartition des segments en SQL

```python
spark.sql("""
    WITH ca_client AS (
        SELECT c.id_client,
               SUM(c.quantite * p.prix) AS ca_total
        FROM commandes c
        JOIN produits p ON c.id_produit = p.id_produit
        WHERE c.statut = 'livree'
        GROUP BY c.id_client
    ),
    segments AS (
        SELECT id_client, ca_total,
               CASE WHEN ca_total >= 25000 THEN 'Gold'
                    WHEN ca_total >= 20000 THEN 'Silver'
                    ELSE 'Bronze'
               END AS segment
        FROM ca_client
    )
    SELECT segment, COUNT(*) AS nb_clients
    FROM segments
    GROUP BY segment
    ORDER BY nb_clients DESC
""").show()
```

```
+-------+----------+
|segment|nb_clients|
+-------+----------+
| Bronze|        22|
|   Gold|        16|
| Silver|        16|
+-------+----------+
```

La requête SQL, avec deux CTE (`ca_client` puis `segments`) et un `CASE WHEN`, donne la même répartition que l'API DataFrame de la question 6.3 : 16 Gold, 16 Silver et 22 Bronze.

### Question 7.2 – Top 5 des clients avec leur segment

```python
spark.sql("""
    WITH ca_client AS (
        SELECT c.id_client,
               SUM(c.quantite * p.prix) AS ca_total
        FROM commandes c
        JOIN produits p ON c.id_produit = p.id_produit
        WHERE c.statut = 'livree'
        GROUP BY c.id_client
    )
    SELECT cl.id_client, cl.nom, cl.ville,
           ROUND(cc.ca_total, 2) AS ca_total,
           CASE WHEN cc.ca_total >= 25000 THEN 'Gold'
                WHEN cc.ca_total >= 20000 THEN 'Silver'
                ELSE 'Bronze'
           END AS segment
    FROM ca_client cc
    JOIN clients cl ON cc.id_client = cl.id_client
    ORDER BY cc.ca_total DESC
    LIMIT 5
""").show()
```

```
+---------+-----------------+------+--------+-------+
|id_client|              nom| ville|ca_total|segment|
+---------+-----------------+------+--------+-------+
|       37|     Hamza Alaoui|   Fès| 41736.0|   Gold|
|       41|Meryem El Idrissi|   Fès| 38512.0|   Gold|
|        6|Khadija Cherkaoui| Rabat| 35376.0|   Gold|
|       43|    Meryem Alaoui|Tanger| 32523.0|   Gold|
|       39| Salma El Idrissi|Agadir| 31485.0|   Gold|
+---------+-----------------+------+--------+-------+
```

La segmentation en SQL (CASE WHEN) donne exactement la même répartition que l’API DataFrame (when/otherwise) : 16 Gold, 16 Silver et 22 Bronze. Les 5 meilleurs clients sont tous Gold, et 2 d’entre eux habitent Fès (Hamza Alaoui et Meryem El Idrissi). Les deux approches sont équivalentes ; on choisit l’une ou l’autre selon la lisibilité.

---

## Partie 8 – Plans d'exécution et optimisation

### Question 8.1 – Lecture d'un plan avec `explain("formatted")`

```python
requete = (commandes
    .join(produits, "id_produit")
    .filter(F.col("categorie") == "Sport"))

requete.explain("formatted")
```

```
== Physical Plan ==
AdaptiveSparkPlan (17)
+- Project (16)
   +- BroadcastHashJoin Inner BuildRight (15)
      :- HashAggregate (5)
      :  +- Exchange (4)
      :     +- HashAggregate (3)
      :        +- Filter (2)
      :           +- Scan csv  (1)
      +- BroadcastExchange (14)
         +- Project (13)
            +- Filter (12)
               +- Window (11)
                  +- Sort (10)
                     +- Exchange (9)
                        +- Project (8)
                           +- Filter (7)
                              +- Scan csv  (6)


...
(6) Scan csv 
Output [4]: [id_produit#0, nom#1, categorie#2, prix#3]
Batched: false
Location: InMemoryFileIndex [file:/C:/Users/Salah/Desktop/TP3_SparkSQL/data_tp2/produits.csv]
PushedFilters: [IsNotNull(categorie), EqualTo(categorie,Sport)]
ReadSchema: struct<id_produit:string,nom:string,categorie:string,prix:string>

(7) Filter
Input [4]: [id_produit#0, nom#1, categorie#2, prix#3]
Condition : (isnotnull(categorie#2) AND (categorie#2 = Sport))


(plan complet dans le notebook ; seuls l'arbre et les nœuds 6 et 7 sont reproduits ici)
```

Bien que le filtre categorie = 'Sport' soit écrit après la jointure, l’optimiseur Catalyst l’a déplacé avant : il apparaît juste au-dessus de la lecture de produits.csv (nœud Filter) et dans les PushedFilters du Scan csv. Cette optimisation s’appelle le predicate pushdown, ou poussée des prédicats (filter pushdown). En filtrant le plus tôt possible, Spark ne lit et ne joint que les 4 produits Sport au lieu des 13, et ne transporte que les lignes utiles dans la suite du plan. Catalyst a pu pousser le filtre jusque sous la fonction de fenêtre de la question 3.2, car celle-ci est partitionnée par categorie : filtrer une catégorie ne change pas la moyenne calculée dans cette catégorie. On remarque aussi que la jointure est déjà un BroadcastHashJoin, ce qui prépare la question 8.2.

### Question 8.2 – Stratégies de jointure : SortMergeJoin et broadcast

```python
requete = lambda: commandes.join(produits, "id_produit").filter(F.col("categorie") == "Sport")

# 0) Valeur par défaut du seuil
print("Seuil par défaut :", spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))

# 1) Broadcast automatique désactivé
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
print("\n===== 1) Broadcast automatique désactivé =====")
requete().explain()

# 2) Broadcast forcé manuellement (toujours avec le seuil à -1)
print("\n===== 2) Broadcast forcé avec F.broadcast =====")
(commandes.join(F.broadcast(produits), "id_produit")
 .filter(F.col("categorie") == "Sport")
 .explain())

# 3) Restauration de la configuration d'origine
spark.conf.unset("spark.sql.autoBroadcastJoinThreshold")
print("\nSeuil restauré :", spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))
```

```
Seuil par défaut : 10485760b

===== 1) Broadcast automatique désactivé =====
(extrait du plan)
+- SortMergeJoin [id_produit#314], [id_produit#357], Inner

===== 2) Broadcast forcé avec F.broadcast =====
(extrait du plan)
+- BroadcastHashJoin [id_produit#314], [id_produit#357], Inner, BuildRight, false, false

Seuil restauré : 10485760b
```

Par défaut, Spark choisit un BroadcastHashJoin, car produits (13 lignes) est bien en dessous du seuil de 10 Mo. Quand on désactive la diffusion automatique (-1), Spark passe à un SortMergeJoin : les deux tables sont redistribuées sur le réseau selon id_produit (Exchange), puis triées (Sort) avant d’être fusionnées. Avec F.broadcast(produits), on retrouve le BroadcastHashJoin même avec le seuil à -1, car le hint est prioritaire sur la configuration.

Pourquoi le broadcast est efficace ici : produits est minuscule par rapport à commandes. Spark en envoie une copie complète à chaque exécuteur, qui la garde en mémoire sous forme de table de hachage. Chaque partition de commandes est alors jointe localement, sans déplacer ni trier la grosse table. On évite le shuffle et le tri de commandes, les opérations les plus coûteuses d’une jointure. Le broadcast n’est intéressant que si la petite table tient en mémoire sur chaque exécuteur ; diffuser une grosse table saturerait le réseau et la mémoire.

### Question 8.3 – Effet du cache

```python
import time

def mesurer(label, n=3):
    durees = []
    for i in range(n):
        t0 = time.time()
        ventes.groupBy("ville").agg(F.sum("ca")).collect()
        durees.append(time.time() - t0)
        print(f"{label} – essai {i+1} : {durees[-1]:.3f} s")
    print(f"{label} – moyenne : {sum(durees)/n:.3f} s\n")
    return sum(durees) / n

# 1) Sans cache
moy_sans = mesurer("Sans cache")

# 2) Avec cache
ventes.cache()
t0 = time.time()
ventes.count()                       # 1re action : remplit le cache
print(f"Remplissage du cache : {time.time() - t0:.3f} s\n")

moy_avec = mesurer("Avec cache")

print(f"Gain : {moy_sans / moy_avec:.1f}x plus rapide avec le cache")
print("Niveau de stockage :", ventes.storageLevel)
```

```python
ventes.unpersist()
print("Niveau de stockage après unpersist :", ventes.storageLevel)
```

```
Sans cache – essai 1 : 0.294 s
Sans cache – essai 2 : 0.229 s
Sans cache – essai 3 : 0.263 s
Sans cache – moyenne : 0.262 s

Remplissage du cache : 0.509 s

Avec cache – essai 1 : 0.175 s
Avec cache – essai 2 : 0.123 s
Avec cache – essai 3 : 0.124 s
Avec cache – moyenne : 0.141 s

Gain : 1.9x plus rapide avec le cache
Niveau de stockage : Disk Memory Deserialized 1x Replicated

Niveau de stockage après unpersist : Serialized 1x Replicated
```

Sans cache, chaque exécution recalcule ventes depuis la lecture des fichiers CSV, car Spark évalue les transformations de façon paresseuse. Après cache() et une première action pour remplir le cache, les exécutions suivantes lisent directement les données en mémoire, et le temps moyen passe de 0,262 s à 0,141 s (gain d'environ 1,9×). L’onglet Storage de l’interface Spark (localhost:4040) confirme que ventes est stocké en mémoire. Le gain reste modeste ici, car le jeu de données est petit, mais il devient important quand un DataFrame coûteux à calculer est réutilisé plusieurs fois. En contrepartie, le cache consomme de la mémoire : il faut appeler unpersist() dès qu’il n’est plus utile.

### Question 8.4 – API DataFrame ou SQL ?

```python
import re

# Même requête, deux écritures
df_api = (commandes
    .filter(F.col("statut") == "livree")
    .join(produits, "id_produit")
    .groupBy("categorie")
    .agg(F.sum(F.col("quantite") * F.col("prix")).alias("ca")))

df_sql = spark.sql("""
    SELECT p.categorie, SUM(c.quantite * p.prix) AS ca
    FROM commandes c
    JOIN produits p ON c.id_produit = p.id_produit
    WHERE c.statut = 'livree'
    GROUP BY p.categorie
""")

print("===== Plan API DataFrame =====")
df_api.explain()
print("===== Plan SQL =====")
df_sql.explain()

# Comparaison automatique des plans physiques
# (on retire les identifiants internes #12, plan_id=..., qui changent à chaque exécution)
def plan_physique(df):
    texte = df._jdf.queryExecution().executedPlan().toString()
    return re.sub(r"#\d+|plan_id=\d+", "", texte)

print("Plans physiques identiques :", plan_physique(df_api) == plan_physique(df_sql))
```

```
===== Plan API DataFrame =====
== Physical Plan ==
AdaptiveSparkPlan isFinalPlan=false
+- HashAggregate(keys=[categorie#2], functions=[sum((cast(quantite#315 as double) * prix#359))])
   +- Exchange hashpartitioning(categorie#2, 8), ENSURE_REQUIREMENTS, [plan_id=12513]
      +- HashAggregate(keys=[categorie#2], functions=[partial_sum((cast(quantite#315 as double) * prix#359))])
         +- Project [quantite#315, categorie#2, prix#359]
            +- BroadcastHashJoin [id_produit#314], [id_produit#357], Inner, BuildRight, false, false
   ...

===== Plan SQL =====
(même plan, mêmes opérateurs)

Plans physiques identiques : True

(plans complets dans le notebook)
```

Les plans physiques de la requête écrite en API DataFrame et en SQL sont strictement identiques : même jointure (BroadcastHashJoin), même agrégation en deux temps (partielle puis finale, séparées par un Exchange), même filtre poussé jusqu’à la lecture du fichier. Les deux écritures passent par le même optimiseur Catalyst : elles sont transformées en un plan logique commun, optimisé, puis converti en plan physique exécuté par Tungsten. Il n’y a donc aucune différence de performance entre SQL et API DataFrame. Le choix dépend de la lisibilité et du contexte : SQL est plus naturel pour les analystes et les requêtes déclaratives, l’API DataFrame se prête mieux à la construction dynamique de requêtes, à la réutilisation de code Python, et à la détection de certaines erreurs (noms de colonnes, syntaxe) dans l’éditeur. On peut d’ailleurs mélanger les deux librement, comme on l’a fait en 4.3.

---

## Partie 9 – Écriture et relecture au format Parquet

### Question 9.1 – Écriture partitionnée par mois

```python
import os

# Ajout des colonnes annee et mois (entiers) à partir de la date de commande
ventes_part = (ventes
    .withColumn("annee", F.year("date_commande"))
    .withColumn("mois",  F.month("date_commande"))
)

# Écriture Parquet partitionnée par mois : un sous-dossier mois=N par valeur
(ventes_part.write
    .mode("overwrite")
    .partitionBy("mois")
    .parquet("out_tp2/ventes"))

# Affichage de l'arborescence
for racine, dossiers, fichiers in os.walk("out_tp2/ventes"):
    dossiers.sort()
    niveau = racine.replace("out_tp2/ventes", "").count(os.sep)
    print("    " * niveau + os.path.basename(racine) + "/")
    for f in sorted(fichiers):
        if not f.endswith(".crc"):          # on masque les fichiers de contrôle .crc
            print("    " * (niveau + 1) + f)
```

```
ventes/
    _SUCCESS
    mois=1/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=10/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=11/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=12/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=2/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=3/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=4/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=5/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=6/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=7/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=8/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
    mois=9/
        part-00000-1e4c3726-c373-4ad6-b83b-c715da2a124b.c000.snappy.parquet
```

L'écriture crée dans `out_tp2/ventes` un dossier par valeur de la colonne de partition : `mois=1`, `mois=2`, …, `mois=12`, soit 12 sous-dossiers, plus un fichier `_SUCCESS` qui indique que le job s'est terminé correctement. Chaque dossier contient ici un seul fichier `part-….snappy.parquet`, compressé en Snappy, car le jeu de données est petit (2089 lignes). La colonne `mois` n'est plus stockée dans les fichiers : sa valeur est portée par le nom du dossier, ce qui évite de la répéter dans chaque ligne. Les dossiers apparaissent dans l'ordre alphabétique (`mois=10` avant `mois=2`), ce qui est sans conséquence pour Spark. Le mode `overwrite` permet de relancer la cellule sans erreur « le dossier existe déjà ». Le partitionnement par mois est pertinent car les analyses filtrent souvent sur une période ; il faut en revanche éviter de partitionner sur une colonne à forte cardinalité (par exemple `id_client`), qui produirait une multitude de petits fichiers.

> **Correction de l'erreur Windows.** L'erreur `UnsatisfiedLinkError: NativeIO$Windows.access0` venait du fait que `hadoop.dll` n'était pas chargée par la JVM. Dans la cellule SparkSession, l'option `-Djava.library.path=C:\hadoop\bin` était transmise à la JVM avec ses antislashs, qui ont été supprimés : la JVM voyait `C:hadoopbin` (vérifié avec `System.getProperty("java.library.path")`), et `NativeCodeLoader.isNativeCodeLoaded()` renvoyait `False`. Avec des slashs (`C:/hadoop/bin`) la bibliothèque native est chargée et l'écriture fonctionne. Les fichiers `winutils.exe` et `hadoop.dll` étaient corrects ; seule la cellule SparkSession (Q1.1) a été modifiée, sur cette ligne.

### Question 9.2 – Relecture, partition pruning et pushdown

```python
# Relecture du dossier partitionné : la colonne "mois" est reconstruite depuis les noms de dossiers
ventes_pq = spark.read.parquet("out_tp2/ventes")
ventes_pq.printSchema()

# 1) Filtre sur la colonne de partition (mois)
print("===== Filtre mois = 3 (colonne de partition) =====")
f_mois = ventes_pq.filter(F.col("mois") == 3)
f_mois.explain()
print("Lignes du mois 3 :", f_mois.count())

# 2) Filtre sur une colonne ordinaire (ville), non partitionnée
print("\n===== Filtre ville = 'Casablanca' (colonne non partitionnée) =====")
f_ville = ventes_pq.filter(F.col("ville") == "Casablanca")
f_ville.explain()
print("Lignes de Casablanca :", f_ville.count())

# Extraction automatique des deux lignes utiles du plan (PartitionFilters / PushedFilters)
import re
for label, df in [("mois = 3", f_mois), ("ville = Casablanca", f_ville)]:
    plan = df._jdf.queryExecution().executedPlan().toString()
    print(f"\n[{label}]")
    for cle in ("PartitionFilters", "PushedFilters"):
        m = re.search(cle + r": \[.*?\]", plan)
        print("  ", m.group(0) if m else cle + " : absent")
```

```
root
 |-- id_commande: integer (nullable = true)
 |-- date_commande: date (nullable = true)
 |-- nom_client: string (nullable = true)
 |-- ville: string (nullable = true)
 |-- id_client: integer (nullable = true)
 |-- nom: string (nullable = true)
 |-- categorie: string (nullable = true)
 |-- quantite: integer (nullable = true)
 |-- prix: double (nullable = true)
 |-- ca: double (nullable = true)
 |-- annee: integer (nullable = true)
 |-- mois: integer (nullable = true)

===== Filtre mois = 3 (colonne de partition) =====
Lignes du mois 3 : 189
===== Filtre ville = 'Casablanca' (colonne non partitionnée) =====
Lignes de Casablanca : 291
[mois = 3]
   PartitionFilters: [isnotnull(mois#2616), (mois#2616 = 3)]
   PushedFilters: []
[ville = Casablanca]
   PartitionFilters: []
   PushedFilters: [IsNotNull(ville), EqualTo(ville,Casablanca)]

(les deux plans `explain()` complets figurent dans le notebook)
```

Les deux plans montrent deux mécanismes différents.

- **Filtre `mois = 3` (colonne de partition)** : le plan contient `PartitionFilters: [isnotnull(mois), (mois = 3)]` et `PushedFilters: []`. Spark utilise le **partition pruning** : il élimine les dossiers `mois=1`, `mois=2`, `mois=4`… dès la planification, et ne lit que `mois=3` (189 lignes). Aucun fichier des autres mois n'est ouvert et il n'y a même plus de nœud `Filter` dans le plan, puisque tout le dossier lu satisfait déjà la condition.
- **Filtre `ville = 'Casablanca'` (colonne ordinaire)** : `PartitionFilters: []`, donc aucun dossier n'est écarté, les 12 sont parcourus. En revanche, `PushedFilters: [IsNotNull(ville), EqualTo(ville,Casablanca)]` : c'est le **pushdown de prédicat**. Le filtre est transmis au lecteur Parquet, qui peut utiliser les statistiques min/max de chaque bloc pour sauter ceux qui ne peuvent pas contenir « Casablanca ». Le nœud `Filter` reste dans le plan, car Parquet ne garantit pas de renvoyer uniquement les lignes exactes : Spark revérifie la condition (291 lignes retenues).

La différence est donc la suivante : le partition pruning écarte des **fichiers et dossiers entiers** sans les ouvrir, grâce à l'organisation sur disque, tandis que le pushdown réduit seulement ce qui est lu **à l'intérieur** des fichiers, après leur ouverture. Le pruning est généralement bien plus efficace, mais il n'est possible que sur la colonne choisie pour `partitionBy`, d'où l'intérêt de choisir cette colonne selon les filtres les plus fréquents.

### Question 9.3 – Rapport final : un CSV unique

```python
import glob

# CA par catégorie et par mois, trié pour un rapport lisible
rapport = (ventes_part
    .groupBy("categorie", "mois")
    .agg(F.round(F.sum("ca"), 2).alias("ca"))
    .orderBy("categorie", "mois"))

# coalesce(1) : une seule partition en sortie, donc un seul fichier part-*.csv
(rapport.coalesce(1)
    .write.mode("overwrite")
    .option("header", True)
    .csv("out_tp2/rapport_ca"))

# Lecture du fichier avec Python (sans passer par Spark)
fichiers = glob.glob("out_tp2/rapport_ca/part-*.csv")
print("Fichiers part-*.csv :", [os.path.basename(f) for f in fichiers])
with open(fichiers[0], encoding="utf-8") as f:
    contenu = f.read()
print(contenu)
print("Nombre de lignes de données :", len(contenu.strip().splitlines()) - 1)
```

```
Fichiers part-*.csv : ['part-00000-15bb94d3-4cf0-4a98-8eab-a1ba1e61f8e9-c000.csv']
categorie,mois,ca
Informatique,1,35488.0
Informatique,2,26147.0
Informatique,3,29241.0
Informatique,4,27263.0
Informatique,5,30193.0
Informatique,6,24902.0
Informatique,7,31906.0
Informatique,8,42962.0
Informatique,9,25664.0
Informatique,10,30399.0
Informatique,11,25374.0
Informatique,12,24772.0
Maison,1,26724.0
Maison,2,24820.0
Maison,3,25377.0
Maison,4,17389.0
Maison,5,21075.0
Maison,6,26957.0
Maison,7,27648.0
Maison,8,18989.0
Maison,9,20767.0
Maison,10,20265.0
Maison,11,25353.0
Maison,12,30761.0
Sport,1,55805.0
Sport,2,47823.0
Sport,3,54833.0
Sport,4,36801.0
Sport,5,35519.0
Sport,6,62393.0
Sport,7,46527.0
Sport,8,52051.0
Sport,9,45646.0
Sport,10,39753.0
Sport,11,48663.0
Sport,12,48873.0

Nombre de lignes de données : 36
```

Le rapport contient 36 lignes de données (3 catégories × 12 mois) plus l'en-tête `categorie,mois,ca`. Spark produit un **dossier** `out_tp2/rapport_ca` et non un fichier : on y trouve un seul `part-….csv` (grâce à `coalesce(1)`), accompagné de `_SUCCESS`, ce qui explique pourquoi on le lit avec `glob("part-*.csv")` car son nom contient un identifiant généré. La lecture en Python confirme les ordres de grandeur déjà observés : par exemple, Sport au mois 1 vaut 55 805 et reste la catégorie dominante dans la plupart des mois. L'option `header` ajoute la ligne de titres, qui est absente par défaut. Un fichier CSV unique est pratique pour un livrable destiné à un humain (Excel, e-mail), ce qui justifie ici le recours à `coalesce(1)`.

### Question 9.4 – Pourquoi utiliser `coalesce(1)` avec précaution ?

`coalesce(1)` ramène toutes les données dans **une seule partition**, donc traitée par **une seule tâche sur un seul cœur** : l'écriture n'est plus parallèle, ce qui supprime l'intérêt d'un moteur distribué. Sur de gros volumes, cela provoque trois problèmes :

1. **Performance** : une seule tâche écrit tout le résultat, alors que les autres cœurs ou machines restent inactifs. Comme `coalesce` ne déclenche pas de shuffle, le calcul en amont (jointures, agrégations) peut lui-même être exécuté par cette seule tâche. `repartition(1)` évite ce défaut en gardant le calcul parallèle, au prix d'un shuffle.
2. **Mémoire et fiabilité** : toutes les données passent par un seul exécuteur ou driver, avec un risque de saturation mémoire (`OutOfMemoryError`) ou de disque. Si cette tâche échoue, tout est à refaire.
3. **Aval** : un énorme fichier unique est difficile à relire en parallèle, car un fichier CSV compressé n'est pas toujours découpable.

On réserve donc `coalesce(1)` aux petits résultats finaux, comme ce rapport de 36 lignes, destinés à être lus par une personne. Pour les gros volumes, on laisse Spark écrire plusieurs fichiers (ou on réduit le nombre de partitions à une valeur raisonnable, par exemple `coalesce(10)`), et si un fichier unique est indispensable, on le fusionne après coup en dehors de Spark.

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
| Ville au CA le plus élevé / le plus faible | Rabat (253 369) / Casablanca (153 441) |
| Clients sans commande | 6 (id 55 à 60) |
| Segmentation des clients | 16 Gold, 16 Silver, 22 Bronze |
| Plans Spark (SQL vs API DataFrame) | Identiques (BroadcastHashJoin) |
| Cache | 0,262 s → 0,141 s (≈ 1,9×) |
| Parquet partitionné par mois | 12 dossiers `mois=N`, 189 lignes pour `mois = 3` |
| Rapport final | CSV unique de 36 lignes (3 catégories × 12 mois) |

---

## Conclusion

Ce TP montre qu'il faut **auditer avant d'analyser**. Charger toutes les colonnes en `string`, puis les convertir avec `try_cast`, permet de traiter des données sales sans faire planter Spark en mode ANSI. L'audit a révélé des doublons, des valeurs manquantes, des types invalides et des incohérences de casse et d'espaces. Sans nettoyage, le chiffre d'affaires aurait été faux : 891 commandes livrées auraient été ignorées à cause de la casse du statut, et les commandes du Casque auraient eu un prix `null`. Enfin, les vues temporaires permettent d'exprimer les analyses en SQL avec le même résultat qu'en API DataFrame.

La seconde moitié du TP a porté sur l'exploitation et l'optimisation. Les jointures (`inner`, `left`, `left_semi`, `left_anti`) ont permis de construire `ventes` (2089 lignes) en vérifiant que la somme des CA par ville retombe sur le CA total de 1 215 123, et d'identifier 6 clients sans commande. Les fonctions de fenêtre ont servi au classement des produits, au CA cumulé et à la segmentation (16 Gold, 16 Silver, 22 Bronze), reproduite à l'identique en SQL avec des CTE. L'étude des plans a montré que Catalyst pousse les filtres au plus près de la lecture, que Spark choisit un `BroadcastHashJoin` pour la petite table `produits`, que le cache réduit le temps moyen de 0,262 s à 0,141 s, et que les écritures SQL et DataFrame produisent le même plan physique. Enfin, l'écriture Parquet partitionnée par mois permet un **partition pruning** (seul le dossier `mois=3` est lu) alors qu'un filtre sur `ville` ne bénéficie que d'un **pushdown** dans les fichiers ; le rapport final est exporté en un CSV unique grâce à `coalesce(1)`, solution réservée aux petits résultats.

**Note sur l'environnement Windows.** L'écriture Parquet échouait avec `UnsatisfiedLinkError: NativeIO$Windows.access0`. La cause était l'option `-Djava.library.path=C:\hadoop\bin` de la cellule SparkSession : les antislashs étaient perdus et la JVM recevait `C:hadoopbin`, si bien que `hadoop.dll` n'était jamais chargée. Écrire `C:/hadoop/bin` avec des slashs a suffi à corriger le problème.
