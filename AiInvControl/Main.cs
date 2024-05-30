using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using Python.Runtime;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System;
using System.IO;
using Microsoft.ML;
using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;
using Microsoft.ML.Data;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AiInvControl
{
    public partial class Main : Form
    {
        private DataTable salesData;
        private InferenceSession session;
        private dynamic labelEncoder;



        public Main()
        {
            InitializeComponent();
            salesData = LoadCsvData("Cafe_12MonthSales.csv");
            LoadModelAndEncoder();


        }
        private void LoadModelAndEncoder()
        {
            session = new InferenceSession("C:/Users/Wence Dante/source/repos/AiInvControl/AiInvControl/sales_prediction_model.onnx");

            using (Py.GIL())
            {
                dynamic joblib = Py.Import("joblib");
                labelEncoder = joblib.load("C:/Users/Wence Dante/source/repos/AiInvControl/AiInvControl/label_encoder.joblib");
            }
        }

        public class SalesData
        {
            public string Cons_Item { get; set; }
            public float Sales { get; set; }
        }
        
  

        private DataTable LoadCsvData(string filePath)
        {
            DataTable dataTable = new DataTable();
            try
            {
                using (StreamReader sr = new StreamReader(filePath))
                {
                    string[] headers = sr.ReadLine().Split(',');
                    foreach (string header in headers) { dataTable.Columns.Add(header); }
                    while (!sr.EndOfStream)
                    {
                        string[] rows = sr.ReadLine().Split(',');
                        DataRow dataRow = dataTable.NewRow();
                        for (int i = 0; i < headers.Length; i++) { dataRow[i] = rows[i]; }
                        dataTable.Rows.Add(dataRow);
                    }
                }
            }
            catch (Exception ex) { MessageBox.Show("Error loading CSV data: " + ex.Message); }
            return dataTable;
        }

        private void guna2vSeparator1_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }
        
        private void product1_Click(object sender, EventArgs e)
        {
            ShowProductDetails("Espresso");
        }

        private void product2_Click(object sender, EventArgs e)
        {
            ShowProductDetails("Cappucino");
        }

        private void product3_Click(object sender, EventArgs e)
        {
            ShowProductDetails("Iced Tea");
        }

        private void product4_Click(object sender, EventArgs e)
        {
            ShowProductDetails("Machiatto");
        }

        private void detailsShow_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }

        private void detailsBtn_Click(object sender, EventArgs e)
        {

        }
        private void ShowProductDetails(string productName)
        {
            DataView dv = salesData.DefaultView;
            dv.RowFilter = $"Cons_Item = '{productName}'";
            detailsShow.DataSource = dv.ToTable();
        }

        private void gunaChart1_Load(object sender, EventArgs e)
        {

        }

        private void guna2Button2_Click(object sender, EventArgs e)
        {
            // Define a list of products to predict sales for
            List<string> products = new List<string> { "Espresso", "Cappucino", "Iced Tea", "Machiatto" };

            // Create a result string to display predictions
            string result = "Predicted Sales:\n";

            foreach (var product in products)
            {
                float predictedSales = PredictSales(product);
                result += $"{product}: {predictedSales}\n";
            }

            // Display the result in a message box
            MessageBox.Show(result);

        }

        private float PredictSales(string productName)
        {
            float predictedSales = 0.0f;
            try
            {
                using (Py.GIL())
                {
                    var encodedProduct = labelEncoder.transform(new string[] { productName });
                    var inputTensor = new DenseTensor<float>(new[] { (float)encodedProduct[0] }, new[] { 1, 1 });

                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("float_input", inputTensor)
                    };

                    using (var results = session.Run(inputs))
                    {
                        var output = results.First().AsEnumerable<float>().ToArray();
                        predictedSales = output[0];
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error predicting sales: {ex.Message}");
            }
            return predictedSales;
        }


        private void saleschart_Load(object sender, EventArgs e)
        {

        }
      
       
    }
}


    


