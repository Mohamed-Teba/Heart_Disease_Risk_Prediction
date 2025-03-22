class HealthRiskExpert(KnowledgeEngine):
    result = "No Risk Detected"
    predicted_slope = None
    predicted_target = None
    
    @Rule(Fact(cp=3) & Fact(thal=3))
    def high_risk(self):
        self.result = "High Risk detected!"
        self.predicted_target = 1
        self.predicted_slope = 2
    
    @Rule(Fact(oldpeak=P(lambda x: x > 2.5)))
    def high_oldpeak(self):
        self.result = "Warning: High oldpeak value detected!"
        self.predicted_target = 1
        self.predicted_slope = 1
    
    @Rule(Fact(exang=1) & Fact(thalach=P(lambda x: x < 100)))
    def exang_thalach_risk(self):
        self.result = "Moderate Risk: Exang and low thalach detected!"
        self.predicted_target = 1
        self.predicted_slope = 0
    
    @Rule(Fact(cp=2) & Fact(thal=2))
    def moderate_risk(self):
        self.result = "Moderate Risk detected!"
        self.predicted_target = 1
        self.predicted_slope = 1
    
    @Rule(Fact(oldpeak=P(lambda x: x < 0)))
    def low_oldpeak(self):
        self.result = "Low Risk: Negative oldpeak value detected!"
        self.predicted_target = 0
        self.predicted_slope = 2
    
    @Rule(Fact(thalach=P(lambda x: x > 180)))
    def high_thalach(self):
        self.result = "High Risk: Extremely high thalach detected!"
        self.predicted_target = 1
        self.predicted_slope = 2
    
    @Rule(Fact(exang=0) & Fact(oldpeak=P(lambda x: x < 1)))
    def no_exang_low_oldpeak(self):
        self.result = "Low Risk: No Exang and low oldpeak!"
        self.predicted_target = 0
        self.predicted_slope = 2
    
    @Rule(Fact(cp=1) & Fact(thalach=P(lambda x: x < 120)))
    def cp1_low_thalach(self):
        self.result = "Moderate Risk: CP type 1 and low thalach detected!"
        self.predicted_target = 1
        self.predicted_slope = 1
    
    @Rule(Fact(oldpeak=P(lambda x: 1 <= x <= 2)))
    def moderate_oldpeak(self):
        self.result = "Moderate Risk: Oldpeak in moderate range!"
        self.predicted_target = 1
        self.predicted_slope = 1
    
    @Rule(Fact(cp=0) & Fact(exang=0) & Fact(thalach=P(lambda x: x > 160)))
    def low_risk_no_cp_exang(self):
        self.result = "Low Risk: No CP, no Exang, and high thalach!"
        self.predicted_target = 0
        self.predicted_slope = 2