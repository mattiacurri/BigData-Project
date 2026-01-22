Ecco un'estrazione dettagliata e strutturata di tutte le informazioni contenute nel documento, separando il testo stampato dagli appunti scritti a mano e dai diagrammi.

### 1. Dataset Originale e Struttura

Il progetto inizia con un "Dataset Originale" reale che contiene informazioni sulle attività degli utenti e le loro interazioni.

* **Struttura dei Dati:**
* Il dataset è composto da post. Ogni riga rappresenta un post.


* I dati sono tabellari (CSV) con le colonne: `post_id`, `user_id`, `content` (testo del post), e `timestamp`.


* 
**Esempio di dati:** Vengono mostrati utenti come Bob, Frank, Raf e Teo con post datati dal 2018 al 2025.




* **La Rete (Grafo):**
* Dai dati si estrae una "Edge list" (lista di archi) che rappresenta un grafo di interazione utente-utente.


* 
*Nota manoscritta:* "Se due utenti sono collegati lo sono per sempre" (implica che le connessioni sono cumulative o statiche una volta formate).




* **Evoluzione Temporale (Snapshots):**
* Il dataset originale viene suddiviso temporalmente per un task di "evoluzione temporale".


* I dati sono divisi nei seguenti intervalli (Snapshots):
1. 2016 - 2021 


2. 2022 


3. 2023 


4. 2024 


5. Gennaio 2025 - Luglio 2025 


6. Luglio 2025 in poi.







### 2. Pre-processing e Embedding

Per preparare i dati all'apprendimento automatico, vengono eseguite diverse operazioni sugli snapshot.

* **Organizzazione:**
* Ogni snapshot è organizzato in cartelle contenenti il CSV dei post di quel periodo.




* **Embedding (Rappresentazione Vettoriale):**
* Viene utilizzato **BERT** per creare embedding del testo dei post.


* 
*Nota manoscritta:* L'embedding dell'utente è calcolato come la **media degli embedding** dei suoi post. L'autore nota: "devo farlo io mi sa".




* **Costruzione delle Connessioni:**
* 
*Assunzione:* Una connessione tra utenti in uno snapshot esiste se l'utente ha post in tutti gli snapshot considerati fino a quello corrente.


* Tuttavia, un'altra nota specifica "Solo connessioni nello snapshot", suggerendo che per il training incrementale potrebbero contare solo le interazioni attive nel periodo.





### 3. Metodologie di Training (Link Prediction)

Il documento contrappone due approcci per la predizione dei link: **Batch** e **Incremental**.

#### A. Temporal Link Prediction (Batch)

* 
**Metodologia:** Utilizza l'algoritmo **GraphSAGE** (citando il paper arXiv:1706.02216).


* **Processo:**
1. Apprendere un modello su uno snapshot .


2. Applicare il modello allo snapshot successivo  per misurare le performance.


3. Unire i due snapshot ().


4. Ripartire dal punto 1 con il dataset unito.




* **Iterazioni Visive:**
* Iterazione 1: Train()  Test().


* Iterazione 2: Train()  Test().


* ...fino all'Iterazione 5: Train()  Test().




* **Dettagli Tecnici (Note manoscritte):**
* L'architettura prevede: Input Node Embedding  MLP  GraphSAGE Layers  Output.


* "Ogni layer va più in profondità nel grafo".


* 
*Aspettativa:* "Non si aspetta grandi risultati da GraphSAGE".





#### B. Temporal Link Prediction (Incremental)

* 
**Metodologia:** Utilizza l'algoritmo **EvolveGCN** (citando il paper arXiv:1902.10191).


* **Processo:**
1. Apprendere un modello su .


2. Testare su .


3. Aggiungere  e **aggiornare** il modello (Update).


4. Testare su .




* **Iterazioni Visive:**
* Il modello non viene riaddestrato su tutto lo storico, ma fa un "Update" basato sulla *ground truth* del nuovo snapshot.


* Iterazione 1: Train()  Test().
* Iterazione 2: Update()  Test().
* Iterazione 3: Update()  Test(), ecc..




* 
**Note:** Si fa riferimento a "Bias Variance analysis" e correlazione di Spearman. Bisogna "adattare i dati rispetto al codice".



### 4. Metriche e Valutazione

Durante il training (sia batch che incrementale), è fondamentale monitorare specifiche metriche per ogni iterazione.

* **Metriche di Performance:**
* Mean Average Precision (MAP)
* AUC Score
* Precision, Recall, F1.




* **Proprietà della Rete:**
* Bisogna tracciare come cambiano le proprietà strutturali della rete generata: Grado medio (avg. degree), cammino minimo medio (avg shortest path), modularità, clustering medio.


* Si suggerisce l'uso della libreria **NetworkX** per analizzare "come la rete può divenire".





### 5. Applicazione su Dataset Sintetico

Una volta addestrato il modello sul dataset reale, l'obiettivo è applicarlo a un dataset sintetico per generare connessioni tra utenti artificiali.

* **Caratteristiche del Dataset Sintetico:**
* 1000 utenti creati artificialmente.


* 10 post per utente generati con un LLM (Large Language Model).


* Nessuna informazione temporale nativa, ma viene simulata una divisione in 3 snapshot ( connessioni iniziali).




* **Flusso di Lavoro (Transfer Learning):**
1. Si prende il modello addestrato sull'ultimo snapshot del dataset reale ().


2. 
**Inferenza:** Si applica questo modello allo "SNAP 1" del dataset sintetico per predire i link.


3. 
**Aggiornamento:** Il modello viene ri-addestrato (batch) o aggiornato (incremental) con i nuovi dati sintetici.


4. Si ripete l'inferenza sullo "SNAP 2" e poi sullo "SNAP 3".




* **Obiettivo Finale:**
* Non c'è uno step di valutazione sul sintetico (poiché non c'è ground truth).


* "Alla fine ho una grande rete".


* Vengono esclusi i nodi usati per il training (quelli reali) e si mantiene solo la rete formata dai nodi del dataset sintetico.