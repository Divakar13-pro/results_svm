pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                echo "Installing dependencies..."
                sh 'python3 -m venv venv'
                sh '. venv/bin/activate && pip install --upgrade pip'
                sh '. venv/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                echo "Running Decision Tree and Random Forest..."
                sh '. venv/bin/activate && python decision_tree_random_forest.py'
            }
        }

        stage('Evaluate Performance') {
            steps {
                echo "Evaluating model performance..."
                // You can modify your Python script to save metrics to results.txt
                sh 'cat results.txt || echo "No results file generated."'
            }
        }
    }

    post {
        always {
            echo "Archiving artifacts..."
            archiveArtifacts artifacts: '*.png, results.txt', allowEmptyArchive: true
        }
    }
}
