import  Footer  from "./Footer";
import  Hero  from "./Hero";
export const LandingPage = () => {
  return (
    <div className="flex flex-col min-h-screen">
      <main className="flex-grow">
        <Hero />
      </main>
      <Footer />
    </div>
  );
};
