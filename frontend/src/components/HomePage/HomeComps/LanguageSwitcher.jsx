import { useState, useEffect } from "react";

const LanguageSwitcher = () => {
  const [currentLanguage, setCurrentLanguage] = useState(0);
  const [fade, setFade] = useState(true); 

  const languages = [
    { text: "উত্তর", name: "Bengali" },
    { text: "Ans", name: "English" },
    { text: "उत्तर", name: "Hindi" },
    { text: "ਜਵਾਬ", name: "Punjabi" },
    { text: "جواب", name: "Urdu" },
    { text: "جواب", name: "Arabic" },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      // fade out
      setFade(false);

      setTimeout(() => {
        setCurrentLanguage((prev) => (prev + 1) % languages.length);
        setFade(true); // fade in
      }, 300); // should match transition duration
    }, 1500);

    return () => clearInterval(interval);
  }, []);

  return (
    <span
      className={`text-black min-w-[120px] text-center transition-all duration-300 ${
        fade ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-2"
      }`}
    >
      {languages[currentLanguage].text}
    </span>
  );
};

export default LanguageSwitcher;
