import java.util.*;

public class Main {

    // Sözləri ayırmaq və yalnız 2 və ya daha çox hərfdən ibarət olanları seçmək
    private static List<String> tokenize(String text) {
        String[] words = text.toLowerCase().split("\\W+");
        List<String> tokens = new ArrayList<>();
        for (String w : words) {
            if (w.length() >= 2) {
                tokens.add(w);
            }
        }
        return tokens;
    }

    public static void main(String[] args) {
        String[] docs = {
                "python proqramlasdirma dili ve kod yazmaq",
                "javascript veb proqramlasdirma dili",
                "sql verilenler bazasi ve sorgular",
                "python ve sql proqramlasdirma"
        };

        String query = "python proqramlasdirma";

        // 1. Sənədlərin tokenizasiyası və lüğətin (vocabulary) yaradılması
        List<List<String>> docTokens = new ArrayList<>();
        Set<String> vocabulary = new HashSet<>();
        Map<String, Integer> df = new HashMap<>(); // Document Frequency

        for (String doc : docs) {
            List<String> tokens = tokenize(doc);
            docTokens.add(tokens);

            Set<String> uniqueTokens = new HashSet<>(tokens);
            for (String token : uniqueTokens) {
                vocabulary.add(token);
                df.put(token, df.getOrDefault(token, 0) + 1);
            }
        }

        int N = docs.length;

        // 2. IDF (Inverse Document Frequency) hesablanması
        Map<String, Double> idf = new HashMap<>();
        for (String term : vocabulary) {
            double idfVal = Math.log((1.0 + N) / (1.0 + df.get(term))) + 1.0;
            idf.put(term, idfVal);
        }

        List<String> vocabList = new ArrayList<>(vocabulary);
        double[][] tfidfMatrix = new double[N][vocabList.size()];

        // 3. Sənədlər üçün TF-IDF matrisinin yaradılması və L2 normallaşdırılması
        for (int i = 0; i < N; i++) {
            List<String> tokens = docTokens.get(i);
            Map<String, Integer> tf = new HashMap<>();
            for (String t : tokens) {
                tf.put(t, tf.getOrDefault(t, 0) + 1);
            }

            double norm = 0.0;
            for (int j = 0; j < vocabList.size(); j++) {
                String term = vocabList.get(j);
                if (tf.containsKey(term)) {
                    double val = tf.get(term) * idf.get(term);
                    tfidfMatrix[i][j] = val;
                    norm += val * val;
                }
            }

            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (int j = 0; j < vocabList.size(); j++) {
                    tfidfMatrix[i][j] /= norm;
                }
            }
        }

        // 4. Sorğu (query) üçün TF-IDF vektorunun yaradılması və L2 normallaşdırılması
        List<String> queryTokens = tokenize(query);
        Map<String, Integer> queryTf = new HashMap<>();
        for (String t : queryTokens) {
            queryTf.put(t, queryTf.getOrDefault(t, 0) + 1);
        }

        double[] queryVector = new double[vocabList.size()];
        double queryNorm = 0.0;

        for (int j = 0; j < vocabList.size(); j++) {
            String term = vocabList.get(j);
            if (queryTf.containsKey(term)) {
                double val = queryTf.get(term) * idf.get(term);
                queryVector[j] = val;
                queryNorm += val * val;
            }
        }

        queryNorm = Math.sqrt(queryNorm);
        if (queryNorm > 0) {
            for (int j = 0; j < vocabList.size(); j++) {
                queryVector[j] /= queryNorm;
            }
        }

        // 5. Kosinus Oxşarlığının (Cosine Similarity) hesablanması və nəticələrin sıralanması
        List<Result> results = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            double dotProduct = 0.0;
            for (int j = 0; j < vocabList.size(); j++) {
                dotProduct += tfidfMatrix[i][j] * queryVector[j];
            }
            results.add(new Result(i, dotProduct));
        }

        // Oxşarlığa görə çoxdan aza doğru sıralama
        results.sort(Collections.reverseOrder());

        // Nəticələrin ekrana yazdırılması
        System.out.println("Sorğu: " + query);
        System.out.println("\nNəticələr:\n");

        for (Result r : results) {
            System.out.printf(Locale.US, "Sənəd %d: %s -> uyğunluq = %.3f\n",
                    (r.index + 1), docs[r.index], r.similarity);
        }
    }

    // Nəticələri saxlamaq və sıralamaq üçün köməkçi sinif (Main daxilində, amma main metodunun xaricində)
    static class Result implements Comparable<Result> {
        int index;
        double similarity;

        Result(int index, double similarity) {
            this.index = index;
            this.similarity = similarity;
        }

        @Override
        public int compareTo(Result o) {
            return Double.compare(this.similarity, o.similarity);
        }
    }
}