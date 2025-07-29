# 🗳️ Voter Turnout Analyzer

A comprehensive Streamlit application for analyzing voter registration and turnout data with AI-powered insights and interactive visualizations.

![Voter Analysis Dashboard](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)

## 📊 Features

### Core Analysis
- **📈 Comprehensive Voter Analysis** - 8+ interactive charts per dataset
- **🎭 Party Breakdown Analysis** - Registration and turnout by political party
- **👥 Age Demographics** - Generational voting patterns (when data available)
- **🏆 Precinct Performance** - Identify high and low performing areas
- **📮 Voting Method Analysis** - Compare different voting methods
- **📊 Efficiency Metrics** - Registration and participation rates

### Advanced Features
- **🤖 AI-Powered Insights** - Case studies of cities that improved turnout
- **📄 Interactive HTML Export** - Share comprehensive reports
- **🔍 Multi-Dataset Comparison** - Compare multiple elections or jurisdictions
- **📱 Responsive Design** - Works on desktop and mobile
- **🔐 Secure Authentication** - Password-protected access

### Visualizations
- Pie charts, gauge charts, bar charts, histograms
- Performance dashboards and funnel analysis
- Geographic comparison charts
- Party registration and turnout comparisons
- Age demographic breakdowns

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- OpenAI API key (for AI suggestions feature)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/pouty-sandwich/voter-turnout-analyzer.git
cd voter-turnout-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. **Run the application**
```bash
streamlit run voter_ai_pro_app.py
```

5. **Access the app**
- Open your browser to `http://localhost:8501`
- Login with: Username: `Votertrends`, Password: `ygG">pIA"95)wZ3`

## 📂 Data Format

The app automatically detects CSV file formats, but your data should include:

### Required Columns (automatically detected):
- **Precinct identification** (precinct name, district, ward, etc.)
- **Registration totals** (total registered voters)
- **Vote counts** (total votes cast)

### Optional Columns:
- **Party registration** (Democrat, Republican, Non-affiliated)
- **Party vote counts** (votes by party)
- **Voting method** (mail-in, in-person, early voting)
- **Date of birth** (for age analysis)

### Example Data Structure:
```csv
Precinct Name,Registration - Total,Public Count - Total,Vote Method
Ward 1 Precinct 1,1250,750,In-Person
Ward 1 Precinct 2,980,580,Mail-In
...
```

## 🎯 Usage

1. **Upload CSV Files** - Drag and drop voter data files
2. **Automatic Analysis** - App detects columns and generates insights
3. **Interactive Exploration** - Use charts to explore patterns
4. **AI Insights** - Get case studies of successful turnout improvements
5. **Export Reports** - Generate HTML reports for sharing

## 📊 Analysis Types

### Overview Analysis
- Turnout breakdown and rates
- Key metrics and performance indicators
- Registration efficiency estimates

### Performance Analysis
- Precinct-level performance comparison
- High and low performing area identification
- Benchmark comparisons

### Demographic Analysis
- Party registration and turnout patterns
- Age group participation (when available)
- Voting method preferences

### Strategic Insights
- AI-powered case studies of successful cities
- Actionable recommendations
- Implementation strategies

## 🤖 AI Features

The app uses OpenAI's GPT models to provide:
- **Case Studies**: Cities that successfully increased turnout
- **Specific Strategies**: Concrete steps taken by other jurisdictions
- **Applicable Recommendations**: Tailored suggestions based on your data

## 📈 Export Options

- **JSON**: Raw data and analysis results
- **CSV**: Summary statistics and comparisons
- **Interactive HTML**: Complete report with all charts and analysis

## 🔧 Configuration

### Authentication
Modify credentials in the app file:
```python
passwords = ['your_password_here']
names = ["Your Name"]
usernames = ["your_username"]
```

### API Configuration
Set your OpenAI API key in `.env`:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

## 🛡️ Security

- Environment variables for API keys
- Password-protected access
- No sensitive data stored
- Secure file handling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

- Create an [Issue](https://github.com/yourusername/voter-turnout-analyzer/issues) for bugs or feature requests
- Check existing issues before creating new ones
- Provide sample data and error messages for faster resolution

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)
- AI insights from [OpenAI](https://openai.com/)
- Authentication via [streamlit-authenticator](https://github.com/mkhorasani/Streamlit-Authenticator)

---

**Made with ❤️ for better civic engagement and election analysis**
